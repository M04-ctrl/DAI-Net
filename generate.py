#!/usr/bin/env python3
"""
通用多调制信号数据集生成器 v5.5 (信号数据集版)
- 修改：
  1. 调制类型精简至20种
  2. ~~取消信号功率归一化~~ → 改为逐信号RMS=1.0归一化 (v5.4修复)
  3. 时频图参数改为 nperseg=512, noverlap=256, nfft=512
  4. 噪声仅包含高斯(80%)和衰落(20%)，去除脉冲和窄带
  5. 训练数据改为低SNR与纯净信号成对生成，以图像形式保存
  6. 低SNR从[-10, -8, -6, -4, -2]中随机选择
  7. ★ v5.4: 修复STFT频率轴非单调bug (fftshift)
  8. ★ v5.4: RRC脉冲成形(α=0.35)替代矩形脉冲 — 旁瓣从-13dB→-60dB+
  9. ★ v5.4: OFDM IFFT功率缩放修复 (×√N_subcarriers)
  10. ★ v5.4: FSK频率过渡Hanning平滑 — 消除硬切换频谱展宽
  11. ★ v5.4: 信号时域淡入淡出窗
  12. ★ v5.5: 频率中心改为复基带可观测范围 [-fs/2, fs/2)，允许正/负频方向
  13. ★ v5.5: Train/Test/Val 全部保存 low-SNR 与 clean 成对图像
  14. ★ v5.5: metadata 保存完整信号参数，不生成 YOLO/检测框标注
- 单条数据：一张时频图，含5-20个混合信号
- 高SNR参考：直接使用纯净信号（无噪声）
- 低SNR：随机选择[-10, -8, -6, -4, -2]中的一个
- 总样本：12000条信号 × 2 = 24000张图（三个划分均成对）
- 划分：Train(9600) : Test(1200) : Val(1200) = 8:1:1
- 训练/测试/验证集：低SNR和纯净信号成对保存为图像
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import os
from tqdm import tqdm
import json

print("=" * 70)
print("通用多调制信号数据集生成器 v5.5 (成对信号图像版, STFT修复)")
print("20 Modulation Types | 5-20 Signals/Spectrogram | Gaussian + Fading Noise")
print("成对生成: 低SNR(随机) + 纯净信号(无噪声)")
print("=" * 70)


class SignalGenerator:
    """20种调制类型信号生成器（已删除部分类型）"""
    
    def __init__(self, fs=10e6, duration=10e-3, nfft=512):
        self.fs = fs
        self.duration = duration
        self.N = int(fs * duration)
        self.nfft = nfft
        self.t = np.arange(self.N) / fs
        
        # 频率池：复基带可观测范围内的正/负频率槽位，避免 fc 超出 [-fs/2, fs/2) 后折叠混叠
        self.freq_min = -0.45 * self.fs
        self.freq_max = 0.45 * self.fs
        self.freq_pool = np.linspace(self.freq_min, self.freq_max, 45)
        
        # 20种调制类型（删去OQPSK, OFDM_1024, SC_FDMA, 扩频族, 宽带特定）
        self.mod_types = {
            'bpsk_family': ['BPSK', 'QPSK', '8PSK'],
            'qam_family': ['16QAM', '32QAM', '64QAM', '256QAM'],
            'ask_family': ['OOK', 'ASK4', 'ASK8'],
            'fsk_family': ['2FSK', '4FSK', '8FSK', '16FSK', 'GMSK'],
            'ofdm_family': ['OFDM_64', 'OFDM_256'],
            'analog_family': ['FM', 'AM_DSB', 'AM_SSB']
        }
        self.all_mods = sum(self.mod_types.values(), [])
        
    def generate_signal(self, mod_type, fc, bw, t_start_ratio, t_end_ratio):
        """生成特定调制类型的信号 - 带长度对齐保护"""
        t0 = int(t_start_ratio * self.N)
        t1 = int(t_end_ratio * self.N)
        t1 = min(t1, self.N)
        if t1 - t0 < 1000:
            t1 = min(t0 + 5000, self.N)
        
        duration_samples = t1 - t0
        t_seg = np.arange(duration_samples) / self.fs
        
        # 符号率设置
        symbol_rate = min(bw / 2, 1e6)
        sps = max(int(self.fs / symbol_rate), 4)
        num_symbols = duration_samples // sps
        
        # 生成基带信号（内部已确保长度）
        baseband = self._generate_baseband(mod_type, duration_samples, sps, num_symbols, bw)
        
        # 双重保险：确保基带长度严格等于duration_samples
        baseband = self._ensure_length(baseband, duration_samples)
        
        # 上变频
        carrier = np.exp(1j * 2 * np.pi * fc * t_seg)

        # 再次确保长度匹配（防止浮点误差）
        if len(baseband) != len(carrier):
            min_len = min(len(baseband), len(carrier))
            baseband = baseband[:min_len]
            carrier = carrier[:min_len]

        signal_seg = baseband * carrier

        # ★ 功率归一化：每个信号 RMS = 1.0，消除不同调制类型间的巨大功率差异
        #   修复前: BPSK ~0.88, 256QAM ~11.5, OFDM_64 ~0.11（差异达 43 dB）
        #   修复后: 全部 RMS = 1.0，所有信号在混合中贡献相同
        rms = np.sqrt(np.mean(np.abs(signal_seg) ** 2))
        if rms > 1e-12:
            signal_seg = signal_seg / rms

        # 时域平滑窗：减少信号起始/结束处的频谱展宽
        fade_len = min(int(0.02 * len(signal_seg)), 2000)  # 2% 长度或最多2000点
        if fade_len > 10:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            signal_seg[:fade_len] *= fade_in
            signal_seg[-fade_len:] *= fade_out

        # 填充到完整时长
        full_signal = np.zeros(self.N, dtype=complex)
        actual_len = len(signal_seg)
        end_idx = min(t0 + actual_len, self.N)
        full_signal[t0:end_idx] = signal_seg[:end_idx-t0]
        
        info = {
            'mod_type': mod_type,
            'category': self._get_category(mod_type),
            'fc_hz': float(fc),
            'bw_hz': float(bw),
            'freq_low_hz': float(fc - bw / 2),
            'freq_high_hz': float(fc + bw / 2),
            'duration_samples': int(actual_len),
            't_start_sample': int(t0),
            't_end_sample': int(end_idx),
            't_start_ratio': float(t_start_ratio),
            't_end_ratio': float(t_end_ratio),
            'symbol_rate': float(symbol_rate)
        }
        
        return full_signal, info
    
    def _ensure_length(self, arr, target_len):
        """确保数组长度严格等于目标长度"""
        current_len = len(arr)
        if current_len == target_len:
            return arr
        
        if current_len > target_len:
            return arr[:target_len]
        else:
            padding = np.zeros(target_len - current_len, dtype=complex)
            return np.concatenate([arr, padding])
    
    def _generate_baseband(self, mod_type, N, sps, num_symbols, bw):
        """生成基带信号 - RRC脉冲成形抑制频谱旁瓣，不做功率归一化"""

        # ===== PSK家族 (RRC脉冲成形) =====
        if mod_type == 'BPSK':
            symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
            symbols_up = np.zeros(num_symbols * sps, dtype=complex)
            symbols_up[::sps] = symbols
            base = self._pulse_shape(symbols_up, sps)

        elif mod_type == 'QPSK':
            const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            syms = const[np.random.randint(0, 4, num_symbols)]
            symbols_up = np.zeros(num_symbols * sps, dtype=complex)
            symbols_up[::sps] = syms
            base = self._pulse_shape(symbols_up, sps)

        elif mod_type == '8PSK':
            angles = np.linspace(0, 2*np.pi, 9)[:-1] + np.pi/8
            const = np.exp(1j * angles)
            syms = const[np.random.randint(0, 8, num_symbols)]
            symbols_up = np.zeros(num_symbols * sps, dtype=complex)
            symbols_up[::sps] = syms
            base = self._pulse_shape(symbols_up, sps)

        # ===== QAM家族 (RRC脉冲成形) =====
        elif mod_type in ['16QAM', '32QAM', '64QAM', '256QAM']:
            order = int(mod_type.replace('QAM', ''))
            symbols_idx = np.random.randint(0, order, num_symbols)
            syms = self._qam_symbols(symbols_idx, order)
            symbols_up = np.zeros(num_symbols * sps, dtype=complex)
            symbols_up[::sps] = syms
            base = self._pulse_shape(symbols_up, sps)

        # ===== ASK家族 (RRC脉冲成形) =====
        elif mod_type == 'OOK':
            symbols = np.random.randint(0, 2, num_symbols).astype(float)
            symbols_up = np.zeros(num_symbols * sps, dtype=float)
            symbols_up[::sps] = symbols
            base = self._pulse_shape(symbols_up, sps)

        elif mod_type in ['ASK4', 'ASK8']:
            m = int(mod_type.replace('ASK', ''))
            symbols = np.random.randint(0, m, num_symbols)
            levels = np.linspace(-(m-1), m-1, m)
            ask_symbols = levels[symbols]
            symbols_up = np.zeros(num_symbols * sps, dtype=float)
            symbols_up[::sps] = ask_symbols
            base = self._pulse_shape(symbols_up, sps)
            
        # ===== FSK家族 (频率过渡平滑，抑制硬切换频谱展宽) =====
        elif mod_type in ['2FSK', '4FSK', '8FSK', '16FSK']:
            m = int(mod_type.replace('FSK', ''))
            freqs = np.linspace(-bw/3, bw/3, m)
            symbols = np.random.randint(0, m, num_symbols)
            freq_seq = np.repeat(freqs[symbols], sps)
            # 频率过渡平滑: 对频率序列做低通滤波，消除符号间瞬时跳变
            # 使用与 sps 成比例的平滑窗，模拟连续相位 FSK
            smooth_len = max(sps // 4, 4)
            smooth_win = np.hanning(2 * smooth_len + 1)
            smooth_win = smooth_win / np.sum(smooth_win)
            freq_seq = np.convolve(freq_seq, smooth_win, mode='same')
            phase = 2 * np.pi * np.cumsum(freq_seq) / self.fs
            base = np.exp(1j * phase)
            
        elif mod_type == 'GMSK':
            bt = 0.3
            symbols = 2 * np.random.randint(0, 2, num_symbols) - 1
            symbols_up = np.repeat(symbols, sps)
            h = self._gaussian_filter(bt, sps)
            filtered = np.convolve(symbols_up, h, mode='same')
            phase = np.pi * 0.5 * np.cumsum(filtered) / sps
            base = np.exp(1j * phase)
            
        # ===== OFDM家族（仅64和256子载波） =====
        elif mod_type in ['OFDM_64', 'OFDM_256']:
            num_sc = int(mod_type.split('_')[1])
            base = self._ofdm_signal(num_sc, N)
            
        # ===== 模拟家族 =====
        elif mod_type == 'FM':
            mod_f = 10e3
            beta = 5.0
            mod_sig = np.sin(2 * np.pi * mod_f * np.arange(N) / self.fs)
            base = np.exp(1j * beta * mod_sig)
            
        elif mod_type == 'AM_DSB':
            mod_f = 5e3
            m = 0.8
            mod_sig = 1 + m * np.sin(2 * np.pi * mod_f * np.arange(N) / self.fs)
            base = mod_sig
            
        elif mod_type == 'AM_SSB':
            mod_f = 5e3
            m = 0.8
            mod_sig = np.sin(2 * np.pi * mod_f * np.arange(N) / self.fs)
            base = (1 + m * mod_sig) * np.exp(1j * 2 * np.pi * mod_f * np.arange(N) / self.fs)
            
        else:
            # 默认QPSK
            const = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            symbols = const[np.random.randint(0, 4, num_symbols)]
            base = np.repeat(symbols, sps)
        
        # 确保长度一致（不做功率归一化）
        base = self._ensure_length(base, N)
        return base
    
    def _qam_symbols(self, symbols_idx, order):
        """QAM符号映射（不做上采样和归一化）"""
        m = int(np.sqrt(order))
        if m * m == order:
            re = (symbols_idx // m) * 2 - (m - 1)
            im = (symbols_idx % m) * 2 - (m - 1)
            qam_symbols = re + 1j * im
        else:
            re = np.random.randint(-m, m+1, len(symbols_idx))
            im = np.random.randint(-m, m+1, len(symbols_idx))
            qam_symbols = re + 1j * im
        return qam_symbols

    def _qam_modulate(self, symbols, order, sps):
        """QAM调制（不做归一化）—— 保留兼容旧代码"""
        qam_symbols = self._qam_symbols(symbols, order)
        return np.repeat(qam_symbols, sps)
    
    def _rrc_filter(self, sps, rolloff=0.35, span=8):
        """根升余弦 (RRC) 脉冲成形滤波器
        - 替代矩形脉冲，大幅抑制频谱旁瓣（矩形脉冲第一旁瓣仅-13dB，RRC可达-60dB+）
        - 这是解决'纯净信号背景不平整'的关键措施之一
        """
        t = np.arange(-span * sps, span * sps + 1) / sps
        # 处理 t=0 的奇点
        with np.errstate(divide='ignore', invalid='ignore'):
            pi_t_T = np.pi * t
            four_beta_t = 4 * rolloff * t
            # RRC 公式
            h = (np.sin(pi_t_T * (1 - rolloff)) + four_beta_t * np.cos(pi_t_T * (1 + rolloff))) / \
                (pi_t_T * (1 - (four_beta_t) ** 2))
        h[t == 0] = 1 + rolloff * (4 / np.pi - 1)
        # 处理分母为 0 的奇点
        denom_zero = np.abs(1 - (four_beta_t) ** 2) < 1e-10
        t_sing = t[denom_zero]
        h[denom_zero] = (rolloff / np.sqrt(2)) * \
            ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) +
             (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        h = h / np.sqrt(np.sum(h ** 2))  # 归一化能量
        return h

    def _pulse_shape(self, symbols_up, sps, rolloff=0.35):
        """对上采样后的符号序列做 RRC 脉冲成形"""
        h = self._rrc_filter(sps, rolloff)
        shaped = np.convolve(symbols_up, h, mode='same')
        return shaped

    def _gaussian_filter(self, bt, sps):
        """GMSK高斯滤波器"""
        t = np.arange(-3 * sps, 3 * sps + 1) / sps
        h = np.exp(-t**2 * 2 * np.pi**2 * bt**2 / np.log(2))
        return h / np.sum(h)
    
    def _ofdm_signal(self, num_sc, N):
        """OFDM信号生成 - 功率已归一化到单位RMS"""
        num_symbols = N // (num_sc + num_sc // 4) + 1
        data = np.random.randint(0, 4, (num_sc, num_symbols))
        const = np.array([1, 1j, -1, -1j])
        symbols = const[data]
        # np.fft.ifft 会除以 num_sc，这里乘以 sqrt(num_sc) 恢复单位功率
        ofdm_blocks = np.fft.ifft(symbols, axis=0) * np.sqrt(num_sc)
        cp = ofdm_blocks[-num_sc // 4:, :]
        ofdm_with_cp = np.vstack([cp, ofdm_blocks])
        result = ofdm_with_cp.T.flatten()
        return self._ensure_length(result, N)
    
    def _get_category(self, mod_type):
        """获取调制类型所属家族"""
        for cat, mods in self.mod_types.items():
            if mod_type in mods:
                return cat
        return 'unknown'

    def _sample_center_frequency(self, bw, available_freqs):
        """在复基带可观测频带内采样中心频率，保留正/负频方向。"""
        edge_guard = max(bw / 2, 0.1e6) + 0.1e6
        f_min = -self.fs / 2 + edge_guard
        f_max = self.fs / 2 - edge_guard

        candidates = available_freqs[(available_freqs >= f_min) & (available_freqs <= f_max)]
        if len(candidates) > 0 and np.random.rand() > 0.2:
            return float(candidates[0])

        return float(np.random.uniform(f_min, f_max))
    
    def generate_mixture(self, num_signals=None):
        """生成多信号混合（5-20个信号），不做功率归一化"""
        if num_signals is None:
            num_signals = np.random.randint(5, 21)
        
        mixture = np.zeros(self.N, dtype=complex)
        info_list = []
        
        available_freqs = self.freq_pool.copy()
        np.random.shuffle(available_freqs)
        
        for i in range(num_signals):
            category = np.random.choice(list(self.mod_types.keys()))
            mod_type = np.random.choice(self.mod_types[category])
            
            # 带宽设置（删除了扩频和宽带信号的特例）
            if 'OFDM' in mod_type:
                bw = np.random.uniform(0.5e6, 1.5e6)
            else:
                bw = np.random.uniform(0.1e6, 0.8e6)

            fc = self._sample_center_frequency(bw, available_freqs[i:])
            
            duration_ratio = np.random.uniform(0.3, 1.0)
            t_start = np.random.uniform(0, 1 - duration_ratio)
            t_end = t_start + duration_ratio
            
            sig, info = self.generate_signal(mod_type, fc, bw, t_start, t_end)
            mixture += sig
            info_list.append(info)
        
        # 不再做功率归一化
        return mixture, info_list
    
    def add_realistic_noise(self, signal_clean, target_snr_db):
        """混合噪声：高斯80% + 衰落20%（去除脉冲和窄带）"""
        signal_power = np.mean(np.abs(signal_clean)**2)
        noise_power = signal_power / (10 ** (target_snr_db / 10))
        
        N = self.N
        noise = np.zeros(N, dtype=complex)
        
        # 高斯噪声 80%
        gaussian_power = 0.8 * noise_power
        noise += np.sqrt(gaussian_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N))
        
        # 衰落噪声 20%
        fading_power = 0.2 * noise_power
        phase = np.exp(1j * np.random.uniform(-np.pi/6, np.pi/6, N))
        noise += np.sqrt(fading_power / 2) * (np.random.randn(N) + 1j * np.random.randn(N)) * phase
        
        # 确保噪声功率准确
        current_power = np.mean(np.abs(noise)**2)
        if current_power > 0:
            noise = noise * np.sqrt(noise_power / current_power)
        
        return signal_clean + noise, noise
    
    def compute_spectrogram(self, iq_data, flatten_background=True):
        """计算时频图 (nperseg=512, noverlap=256, nfft=512)

        Parameters
        ----------
        iq_data : ndarray
            复基带信号
        flatten_background : bool
            逐频点减除背景中位数，消除频率相关的底噪偏置（水平条纹）

        Returns
        -------
        spec_db : ndarray (n_freq, n_time)
        f : ndarray  频率轴 [Hz]，fftshift 后单调递增 [-fs/2, fs/2)
        t : ndarray  时间轴 [s]
        """
        f, t, Sxx = signal.spectrogram(
            iq_data, fs=self.fs, window='hann',
            nperseg=512, noverlap=256, nfft=self.nfft,
            mode='complex'
        )
        # fftshift: 频率轴单调递增 [-fs/2, fs/2)
        Sxx = np.fft.fftshift(Sxx, axes=0)
        f = np.fft.fftshift(f)
        spec_db = 20 * np.log10(np.abs(Sxx) + 1e-10)

        if flatten_background:
            # ★ 逐频点背景减除: 每个频率bin减去其时间轴中位数
            #    消除 FFT scalloping loss、窗函数频响、信号旁瓣叠加
            #    造成的频率相关底噪偏置（视觉上的水平条纹）
            bg_per_freq = np.median(spec_db, axis=1, keepdims=True)
            spec_db = spec_db - bg_per_freq
            # ★ 硬阈值削底: 低于 threshold 的能量全部归零
            #    背景 std ~4.0 dB，threshold=6.0 dB 盖住 95%+ 的正向毛刺
            #    保证深度学习模型学到的背景是完美的常量 0，无雪花点
            threshold = 8.0
            spec_db = np.where(spec_db < threshold, 0.0, spec_db)

        return spec_db, f, t


def save_spectrogram_image(spec_db, filename, dynamic_range_db=60):
    """将时频图保存为图像

    背景已通过逐频点减除对齐到 0 dB，所以 vmin=0 严格对应
    viridis 最深色，vmax 取信号峰值（至少 20 dB 保证可见性）。
    """
    vmin = 6.0                   # 切掉背景波动 (~4.27dB std)，保证纯黑背景
    vmax = max(np.max(spec_db), 20)  # 信号至少拉到 20 dB
    plt.figure(figsize=(6, 4), dpi=100)
    plt.imshow(spec_db, aspect='auto', origin='lower', cmap='viridis',
               vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def generate_dataset(total_samples=12000, output_dir='signal_dataset_v5', random_seed=None):
    """生成完整数据集"""

    if random_seed is not None:
        np.random.seed(random_seed)

    train_num = int(total_samples * 0.8)
    test_num = int(total_samples * 0.1)
    val_num = total_samples - train_num - test_num
    
    splits = {
        'train': {'path': f'{output_dir}/train', 'paired': True, 'num': train_num},
        'test': {'path': f'{output_dir}/test', 'paired': True, 'num': test_num},
        'val': {'path': f'{output_dir}/val', 'paired': True, 'num': val_num}
    }
    
    # 为每个划分创建成对目录
    for s in splits.values():
        os.makedirs(s['path'], exist_ok=True)
        os.makedirs(f"{s['path']}/low_snr", exist_ok=True)
        os.makedirs(f"{s['path']}/clean", exist_ok=True)
    
    print(f"\n数据集配置:")
    print(f"  总信号组合: {total_samples}")
    print(f"  Train: {splits['train']['num']} (成对)")
    print(f"  Test:  {splits['test']['num']} (成对)")
    print(f"  Val:   {splits['val']['num']} (成对)")
    print(f"  低SNR: 随机选择 [-10, -8, -6, -4, -2] dB")
    print(f"  高SNR参考: 使用纯净信号（无噪声）")
    print(f"  调制: 20种")
    print(f"  每张图: 5-20个信号")
    print(f"  频率方向: 复基带正/负频率均可")
    print(f"  保存格式: PNG图像 + metadata.json（不生成检测框标注）")
    
    gen = SignalGenerator()
    low_snr_levels = [-10, -8, -6, -4, -2]  # 低SNR候选列表
    sample_idx = 0
    metadata_all = []
    
    for split_name, split_info in splits.items():
        print(f"\n{'='*60}")
        print(f"生成 {split_name.upper()}: {split_info['num']}条")
        print(f"{'='*60}")
        
        paired = split_info['paired']
        save_path = split_info['path']
        
        for i in tqdm(range(split_info['num']), desc=split_name):
            mixture_clean, sig_info = gen.generate_mixture()
            
            # 随机选择一个低SNR
            low_snr = int(np.random.choice(low_snr_levels))
            
            # 生成低SNR信号和时频图
            mixture_low_snr, _ = gen.add_realistic_noise(mixture_clean, low_snr)
            spec_low_snr, f_axis, t_axis = gen.compute_spectrogram(mixture_low_snr)
            
            # 保存低SNR图像
            low_snr_filename = f"{save_path}/low_snr/sample_{sample_idx:05d}_SNR{low_snr}dB.png"
            save_spectrogram_image(spec_low_snr, low_snr_filename)
            
            # 使用纯净信号作为高SNR参考（无噪声）
            spec_clean, _, _ = gen.compute_spectrogram(mixture_clean)

            # 保存纯净信号图像
            clean_filename = f"{save_path}/clean/sample_{sample_idx:05d}_clean.png"
            save_spectrogram_image(spec_clean, clean_filename)
            
            metadata_all.append({
                'sample_idx': int(sample_idx),
                'pair_id': f"sample_{sample_idx:05d}",
                'split': split_name,
                'low_snr_db': int(low_snr),
                'low_snr_filename': low_snr_filename,
                'clean_filename': clean_filename,
                'num_signals': int(len(sig_info)),
                'mods': [s['mod_type'] for s in sig_info],
                'signals': sig_info,
                'spectrogram_shape': [int(spec_low_snr.shape[0]), int(spec_low_snr.shape[1])],
                'freq_axis_hz': [float(f_axis[0]), float(f_axis[-1])],
                'time_axis_s': [float(t_axis[0]), float(t_axis[-1])]
            })
            
            if (i + 1) % 200 == 0:
                mods = [s['mod_type'] for s in sig_info]
                print(f"\n  样本{i+1}: {len(sig_info)}个信号, 低SNR={low_snr}dB, 类型: {set(mods)}")
            
            sample_idx += 1
    
    with open(f'{output_dir}/metadata.json', 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_name': 'WBSig-style low-SNR paired wideband signal dataset',
            'description': 'Signal-level paired spectrogram dataset. Classes are modulation types. No YOLO/detection-box labels are generated.',
            'total': int(total_samples),
            'splits': {k: int(v['num']) for k, v in splits.items()},
            'low_snr_levels': low_snr_levels,
            'mod_types': gen.all_mods,
            'sample_rate_hz': float(gen.fs),
            'duration_s': float(gen.duration),
            'num_iq_samples': int(gen.N),
            'nfft': int(gen.nfft),
            'frequency_range_hz': [float(-gen.fs / 2), float(gen.fs / 2)],
            'center_frequency_pool_hz': [float(x) for x in gen.freq_pool],
            'image_note': 'Saved PNGs may be rectangular; spectrogram matrix shape is recorded per sample.',
            'total_images': sum(v['num'] * (2 if v['paired'] else 1) for v in splits.values()),
            'samples': metadata_all
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n完成! 保存至: {os.path.abspath(output_dir)}")
    return output_dir


if __name__ == "__main__":
    output = generate_dataset(total_samples=12000, output_dir='signal_dataset_v5')
    #output = generate_dataset(total_samples=100, output_dir='signal_dataset_test')
    print(f"\n生成完成: {output}")
