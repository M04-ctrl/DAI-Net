#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
通用多调制信号数据集生成器 v5.5
（OFDM带宽诊断修改版）

说明：
1. 保留原 v5.5 的整体数据生成流程。
2. 仅修改 OFDM_64 / OFDM_256 的生成逻辑：
   - OFDM_64 / OFDM_256 表示 FFT size；
   - bw 真正用于决定有效子载波数量；
   - 仅激活中心附近的有效子载波；
   - 避免原版本 OFDM 几乎占满整个 Nyquist 频带。
3. 当前仍保留原 compute_spectrogram() 中：
   - 逐频点时间中位数扣除；
   - 8 dB 硬阈值。
   原因：当前先做 OFDM 带宽控制变量实验。
4. 暂时不要直接运行本文件生成完整 12000 组数据。
   当前应由 compare_spectrogram_preprocessing.py 导入本文件进行诊断。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.io as sio
import os
from tqdm import tqdm
import json


print("=" * 70)
print("通用多调制信号数据集生成器 v5.5 (OFDM带宽诊断修改版)")
print("20 Modulation Types | 5-20 Signals/Spectrogram | Gaussian + Fading Noise")
print("成对生成: 低SNR(随机) + 纯净信号(无噪声)")
print("=" * 70)


class SignalGenerator:
    """20种调制类型信号生成器"""

    def __init__(self, fs=10e6, duration=10e-3, nfft=512):
        self.fs = fs
        self.duration = duration
        self.N = int(fs * duration)
        self.nfft = nfft
        self.t = np.arange(self.N) / fs

        # 复基带可观测频率范围
        self.freq_min = -0.45 * self.fs
        self.freq_max = 0.45 * self.fs
        self.freq_pool = np.linspace(
            self.freq_min,
            self.freq_max,
            45
        )

        # 20种调制类型
        self.mod_types = {
            'bpsk_family': [
                'BPSK',
                'QPSK',
                '8PSK'
            ],
            'qam_family': [
                '16QAM',
                '32QAM',
                '64QAM',
                '256QAM'
            ],
            'ask_family': [
                'OOK',
                'ASK4',
                'ASK8'
            ],
            'fsk_family': [
                '2FSK',
                '4FSK',
                '8FSK',
                '16FSK',
                'GMSK'
            ],
            'ofdm_family': [
                'OFDM_64',
                'OFDM_256'
            ],
            'analog_family': [
                'FM',
                'AM_DSB',
                'AM_SSB'
            ]
        }

        self.all_mods = sum(
            self.mod_types.values(),
            []
        )

    # ============================================================
    # 单个信号生成
    # ============================================================

    def generate_signal(
        self,
        mod_type,
        fc,
        bw,
        t_start_ratio,
        t_end_ratio
    ):
        """生成特定调制类型信号"""

        t0 = int(t_start_ratio * self.N)
        t1 = int(t_end_ratio * self.N)

        t1 = min(t1, self.N)

        if t1 - t0 < 1000:
            t1 = min(t0 + 5000, self.N)

        duration_samples = t1 - t0

        t_seg = np.arange(
            duration_samples
        ) / self.fs

        # --------------------------------------------------------
        # 符号率参数：PSK/QAM/ASK 的带宽适配
        # --------------------------------------------------------

        rrc_mod_types = (
            self.mod_types['bpsk_family']
            + self.mod_types['qam_family']
            + self.mod_types['ask_family']
        )

        if mod_type in rrc_mod_types:
            # PSK/QAM/ASK：RRC带宽映射
            rolloff = 0.35
            symbol_rate = min(
                bw / (1.0 + rolloff),
                1e6
            )

        elif mod_type == "GMSK":
            # GMSK：基于14个测试点的经验带宽映射
            # 单位：bw为Hz，symbol_rate为symbols/s
            gmsk_slope = 0.8904
            gmsk_intercept_hz = 0.0208e6

            symbol_rate = (
                bw - gmsk_intercept_hz
            ) / gmsk_slope

            # 仅适用于本次已验证的符号率范围
            symbol_rate = np.clip(
                symbol_rate,
                0.1e6,
                1.0e6
            )

        else:
            # FSK、OFDM、FM、AM暂时不修改
            symbol_rate = min(bw / 2, 1e6)

        sps = max(
            int(self.fs / symbol_rate),
            4
        )

        num_symbols = duration_samples // sps

        # --------------------------------------------------------
        # 生成基带信号
        # --------------------------------------------------------
        baseband = self._generate_baseband(
            mod_type,
            duration_samples,
            sps,
            num_symbols,
            bw
        )

        baseband = self._ensure_length(
            baseband,
            duration_samples
        )

        # --------------------------------------------------------
        # 上变频
        # --------------------------------------------------------
        carrier = np.exp(
            1j
            * 2
            * np.pi
            * fc
            * t_seg
        )

        if len(baseband) != len(carrier):
            min_len = min(
                len(baseband),
                len(carrier)
            )

            baseband = baseband[:min_len]
            carrier = carrier[:min_len]

        signal_seg = (
            baseband
            * carrier
        )

        # --------------------------------------------------------
        # 单信号 RMS 归一化
        # --------------------------------------------------------
        rms = np.sqrt(
            np.mean(
                np.abs(signal_seg) ** 2
            )
        )

        if rms > 1e-12:
            signal_seg = signal_seg / rms

        # --------------------------------------------------------
        # 时域淡入淡出
        # --------------------------------------------------------
        fade_len = min(
            int(
                0.02
                * len(signal_seg)
            ),
            2000
        )

        if fade_len > 10:

            fade_in = np.linspace(
                0,
                1,
                fade_len
            )

            fade_out = np.linspace(
                1,
                0,
                fade_len
            )

            signal_seg[:fade_len] *= fade_in
            signal_seg[-fade_len:] *= fade_out

        # --------------------------------------------------------
        # 填充到完整观测时长
        # --------------------------------------------------------
        full_signal = np.zeros(
            self.N,
            dtype=complex
        )

        actual_len = len(signal_seg)

        end_idx = min(
            t0 + actual_len,
            self.N
        )

        full_signal[
            t0:end_idx
        ] = signal_seg[
            :end_idx - t0
        ]

        # --------------------------------------------------------
        # 元数据
        # --------------------------------------------------------
        info = {
            'mod_type': mod_type,
            'category': self._get_category(
                mod_type
            ),
            'fc_hz': float(fc),
            'bw_hz': float(bw),
            'freq_low_hz': float(
                fc - bw / 2
            ),
            'freq_high_hz': float(
                fc + bw / 2
            ),
            'duration_samples': int(
                actual_len
            ),
            't_start_sample': int(t0),
            't_end_sample': int(end_idx),
            't_start_ratio': float(
                t_start_ratio
            ),
            't_end_ratio': float(
                t_end_ratio
            ),
            'symbol_rate': float(
                symbol_rate
            )
        }

        return full_signal, info

    # ============================================================
    # 长度保护
    # ============================================================

    def _ensure_length(
        self,
        arr,
        target_len
    ):
        """保证数组长度严格等于 target_len"""

        current_len = len(arr)

        if current_len == target_len:
            return arr

        if current_len > target_len:
            return arr[:target_len]

        padding = np.zeros(
            target_len - current_len,
            dtype=complex
        )

        return np.concatenate(
            [
                arr,
                padding
            ]
        )

    # ============================================================
    # 各调制方式基带信号生成
    # ============================================================

    def _generate_baseband(
        self,
        mod_type,
        N,
        sps,
        num_symbols,
        bw
    ):
        """生成基带调制信号"""

        # ========================================================
        # PSK
        # ========================================================

        if mod_type == 'BPSK':

            symbols = (
                2
                * np.random.randint(
                    0,
                    2,
                    num_symbols
                )
                - 1
            )

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=complex
            )

            symbols_up[::sps] = symbols

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        elif mod_type == 'QPSK':

            const = np.array(
                [
                    1 + 1j,
                    -1 + 1j,
                    -1 - 1j,
                    1 - 1j
                ]
            ) / np.sqrt(2)

            syms = const[
                np.random.randint(
                    0,
                    4,
                    num_symbols
                )
            ]

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=complex
            )

            symbols_up[::sps] = syms

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        elif mod_type == '8PSK':

            angles = (
                np.linspace(
                    0,
                    2 * np.pi,
                    9
                )[:-1]
                + np.pi / 8
            )

            const = np.exp(
                1j * angles
            )

            syms = const[
                np.random.randint(
                    0,
                    8,
                    num_symbols
                )
            ]

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=complex
            )

            symbols_up[::sps] = syms

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        # ========================================================
        # QAM
        # ========================================================

        elif mod_type in [
            '16QAM',
            '32QAM',
            '64QAM',
            '256QAM'
        ]:

            order = int(
                mod_type.replace(
                    'QAM',
                    ''
                )
            )

            symbols_idx = np.random.randint(
                0,
                order,
                num_symbols
            )

            syms = self._qam_symbols(
                symbols_idx,
                order
            )

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=complex
            )

            symbols_up[::sps] = syms

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        # ========================================================
        # ASK
        # ========================================================

        elif mod_type == 'OOK':

            symbols = np.random.randint(
                0,
                2,
                num_symbols
            ).astype(float)

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=float
            )

            symbols_up[::sps] = symbols

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        elif mod_type in [
            'ASK4',
            'ASK8'
        ]:

            m = int(
                mod_type.replace(
                    'ASK',
                    ''
                )
            )

            symbols = np.random.randint(
                0,
                m,
                num_symbols
            )

            levels = np.linspace(
                -(m - 1),
                m - 1,
                m
            )

            ask_symbols = levels[
                symbols
            ]

            symbols_up = np.zeros(
                num_symbols * sps,
                dtype=float
            )

            symbols_up[::sps] = ask_symbols

            base = self._pulse_shape(
                symbols_up,
                sps
            )

        # ========================================================
        # FSK
        # ========================================================

        elif mod_type in [
            '2FSK',
            '4FSK',
            '8FSK',
            '16FSK'
        ]:

            m = int(
                mod_type.replace(
                    'FSK',
                    ''
                )
            )

            freqs = np.linspace(
                -bw / 3,
                bw / 3,
                m
            )

            symbols = np.random.randint(
                0,
                m,
                num_symbols
            )

            freq_seq = np.repeat(
                freqs[symbols],
                sps
            )

            smooth_len = max(
                sps // 4,
                4
            )

            smooth_win = np.hanning(
                2 * smooth_len + 1
            )

            smooth_win = (
                smooth_win
                / np.sum(smooth_win)
            )

            freq_seq = np.convolve(
                freq_seq,
                smooth_win,
                mode='same'
            )

            phase = (
                2
                * np.pi
                * np.cumsum(freq_seq)
                / self.fs
            )

            base = np.exp(
                1j * phase
            )

        elif mod_type == 'GMSK':

            bt = 0.3

            symbols = (
                2
                * np.random.randint(
                    0,
                    2,
                    num_symbols
                )
                - 1
            )

            symbols_up = np.repeat(
                symbols,
                sps
            )

            h = self._gaussian_filter(
                bt,
                sps
            )

            filtered = np.convolve(
                symbols_up,
                h,
                mode='same'
            )

            phase = (
                np.pi
                * 0.5
                * np.cumsum(filtered)
                / sps
            )

            base = np.exp(
                1j * phase
            )

        # ========================================================
        # OFDM
        # ========================================================

        elif mod_type in [
            'OFDM_64',
            'OFDM_256'
        ]:

            # OFDM_64 / OFDM_256 表示 FFT size
            nfft = int(
                mod_type.split('_')[1]
            )

            # 关键修改：
            # bw 真正传入 OFDM 生成器，
            # 用于控制激活子载波数量。
            base = self._ofdm_signal(
                nfft=nfft,
                N=N,
                bw=bw
            )

        # ========================================================
        # 模拟调制
        # ========================================================

        elif mod_type == 'FM':

            mod_f = 10e3
            beta = 5.0

            mod_sig = np.sin(
                2
                * np.pi
                * mod_f
                * np.arange(N)
                / self.fs
            )

            base = np.exp(
                1j
                * beta
                * mod_sig
            )

        elif mod_type == 'AM_DSB':

            mod_f = 5e3
            m = 0.8

            mod_sig = (
                1
                + m
                * np.sin(
                    2
                    * np.pi
                    * mod_f
                    * np.arange(N)
                    / self.fs
                )
            )

            base = mod_sig

        elif mod_type == 'AM_SSB':

            mod_f = 5e3
            m = 0.8

            mod_sig = np.sin(
                2
                * np.pi
                * mod_f
                * np.arange(N)
                / self.fs
            )

            base = (
                1
                + m * mod_sig
            ) * np.exp(
                1j
                * 2
                * np.pi
                * mod_f
                * np.arange(N)
                / self.fs
            )

        else:

            const = np.array(
                [
                    1 + 1j,
                    -1 + 1j,
                    -1 - 1j,
                    1 - 1j
                ]
            ) / np.sqrt(2)

            symbols = const[
                np.random.randint(
                    0,
                    4,
                    num_symbols
                )
            ]

            base = np.repeat(
                symbols,
                sps
            )

        base = self._ensure_length(
            base,
            N
        )

        return base

    # ============================================================
    # QAM 工具
    # ============================================================

    def _qam_symbols(
        self,
        symbols_idx,
        order
    ):
        """QAM符号映射"""

        m = int(
            np.sqrt(order)
        )

        if m * m == order:

            re = (
                symbols_idx // m
            ) * 2 - (m - 1)

            im = (
                symbols_idx % m
            ) * 2 - (m - 1)

            qam_symbols = (
                re
                + 1j * im
            )

        else:

            re = np.random.randint(
                -m,
                m + 1,
                len(symbols_idx)
            )

            im = np.random.randint(
                -m,
                m + 1,
                len(symbols_idx)
            )

            qam_symbols = (
                re
                + 1j * im
            )

        return qam_symbols

    def _qam_modulate(
        self,
        symbols,
        order,
        sps
    ):
        """旧接口兼容"""

        qam_symbols = self._qam_symbols(
            symbols,
            order
        )

        return np.repeat(
            qam_symbols,
            sps
        )

    # ============================================================
    # RRC
    # ============================================================

    def _rrc_filter(
        self,
        sps,
        rolloff=0.35,
        span=8
    ):
        """根升余弦滤波器"""

        t = np.arange(
            -span * sps,
            span * sps + 1
        ) / sps

        with np.errstate(
            divide='ignore',
            invalid='ignore'
        ):

            pi_t_T = np.pi * t

            four_beta_t = (
                4
                * rolloff
                * t
            )

            h = (
                np.sin(
                    pi_t_T
                    * (1 - rolloff)
                )
                + four_beta_t
                * np.cos(
                    pi_t_T
                    * (1 + rolloff)
                )
            ) / (
                pi_t_T
                * (
                    1
                    - four_beta_t ** 2
                )
            )

        h[t == 0] = (
            1
            + rolloff
            * (
                4 / np.pi
                - 1
            )
        )

        denom_zero = (
            np.abs(
                1
                - four_beta_t ** 2
            )
            < 1e-10
        )

        h[denom_zero] = (
            rolloff
            / np.sqrt(2)
        ) * (
            (
                1
                + 2 / np.pi
            )
            * np.sin(
                np.pi
                / (
                    4
                    * rolloff
                )
            )
            +
            (
                1
                - 2 / np.pi
            )
            * np.cos(
                np.pi
                / (
                    4
                    * rolloff
                )
            )
        )

        h = (
            h
            / np.sqrt(
                np.sum(
                    h ** 2
                )
            )
        )

        return h

    def _pulse_shape(
        self,
        symbols_up,
        sps,
        rolloff=0.35
    ):
        """RRC脉冲成形"""

        h = self._rrc_filter(
            sps,
            rolloff
        )

        shaped = np.convolve(
            symbols_up,
            h,
            mode='same'
        )

        return shaped

    # ============================================================
    # GMSK 高斯滤波器
    # ============================================================

    def _gaussian_filter(
        self,
        bt,
        sps
    ):

        t = np.arange(
            -3 * sps,
            3 * sps + 1
        ) / sps

        h = np.exp(
            -t ** 2
            * 2
            * np.pi ** 2
            * bt ** 2
            / np.log(2)
        )

        return (
            h
            / np.sum(h)
        )

    # ============================================================
    # OFDM —— 本次重点修改部分
    # ============================================================

    def _ofdm_signal(
        self,
        nfft,
        N,
        bw
    ):
        """
        生成带宽受控的 OFDM 信号。

        参数
        ----
        nfft : int
            OFDM FFT 大小。
            例如：
            OFDM_64  -> 64
            OFDM_256 -> 256

        N : int
            最终需要生成的时域样本数。

        bw : float
            目标占用带宽，单位 Hz。

        原版本问题
        ----------
        原版本将 nfft 个子载波全部填满，
        因此 OFDM 实际占用几乎整个 fs 带宽，
        metadata 中的 bw 并没有真正生效。

        当前修改
        --------
        根据：

            delta_f = fs / nfft

        计算需要激活的子载波数量：

            num_active ≈ bw / delta_f

        只激活中心附近的 num_active 个频率 bin。
        """

        # --------------------------------------------------------
        # 子载波间隔
        # --------------------------------------------------------
        delta_f = (
            self.fs
            / nfft
        )

        # --------------------------------------------------------
        # 根据目标带宽确定激活子载波数
        # --------------------------------------------------------
        num_active = int(
            round(
                bw
                / delta_f
            )
        )

        # 至少两个有效子载波
        num_active = max(
            2,
            num_active
        )

        # 保留两侧保护 bin
        num_active = min(
            num_active,
            nfft - 2
        )

        # --------------------------------------------------------
        # CP
        # --------------------------------------------------------
        cp_len = (
            nfft // 4
        )

        symbol_len = (
            nfft
            + cp_len
        )

        num_ofdm_symbols = (
            N // symbol_len
            + 1
        )

        blocks = []

        # QPSK 星座
        const = np.array(
            [
                1 + 0j,
                0 + 1j,
                -1 + 0j,
                0 - 1j
            ],
            dtype=complex
        )

        # --------------------------------------------------------
        # 逐 OFDM symbol 生成
        # --------------------------------------------------------
        for _ in range(
            num_ofdm_symbols
        ):

            # fftshift 排列下的频域向量
            X_shifted = np.zeros(
                nfft,
                dtype=complex
            )

            data_idx = np.random.randint(
                0,
                4,
                num_active
            )

            qpsk = const[
                data_idx
            ]

            # ----------------------------------------------------
            # 把有效子载波放到 DC 附近
            # ----------------------------------------------------
            start = (
                nfft // 2
                - num_active // 2
            )

            end = (
                start
                + num_active
            )

            X_shifted[
                start:end
            ] = qpsk

            # ----------------------------------------------------
            # fftshift 格式转成 numpy IFFT 格式
            # ----------------------------------------------------
            X_ifft = np.fft.ifftshift(
                X_shifted
            )

            x = np.fft.ifft(
                X_ifft
            )

            # ----------------------------------------------------
            # 每个 OFDM symbol 做 RMS 归一化
            # ----------------------------------------------------
            rms = np.sqrt(
                np.mean(
                    np.abs(x) ** 2
                )
            )

            if rms > 1e-12:
                x = x / rms

            # ----------------------------------------------------
            # 添加循环前缀
            # ----------------------------------------------------
            cp = x[
                -cp_len:
            ]

            block = np.concatenate(
                [
                    cp,
                    x
                ]
            )

            blocks.append(
                block
            )

        signal_out = np.concatenate(
            blocks
        )

        signal_out = self._ensure_length(
            signal_out,
            N
        )

        return signal_out

    # ============================================================
    # 调制族
    # ============================================================

    def _get_category(
        self,
        mod_type
    ):

        for cat, mods in self.mod_types.items():

            if mod_type in mods:
                return cat

        return 'unknown'

    # ============================================================
    # 中心频率采样
    # ============================================================

    def _sample_center_frequency(
        self,
        bw,
        available_freqs
    ):

        edge_guard = max(
            bw / 2,
            0.1e6
        ) + 0.1e6

        f_min = (
            -self.fs / 2
            + edge_guard
        )

        f_max = (
            self.fs / 2
            - edge_guard
        )

        candidates = available_freqs[
            (
                available_freqs
                >= f_min
            )
            &
            (
                available_freqs
                <= f_max
            )
        ]

        if (
            len(candidates) > 0
            and np.random.rand() > 0.2
        ):
            return float(
                candidates[0]
            )

        return float(
            np.random.uniform(
                f_min,
                f_max
            )
        )

    # ============================================================
    # 多信号混合
    # ============================================================

    def generate_mixture(
        self,
        num_signals=None
    ):

        if num_signals is None:
            num_signals = np.random.randint(
                5,
                21
            )

        mixture = np.zeros(
            self.N,
            dtype=complex
        )

        info_list = []

        available_freqs = (
            self.freq_pool.copy()
        )

        np.random.shuffle(
            available_freqs
        )

        for i in range(
            num_signals
        ):

            category = np.random.choice(
                list(
                    self.mod_types.keys()
                )
            )

            mod_type = np.random.choice(
                self.mod_types[
                    category
                ]
            )

            # ----------------------------------------------------
            # 带宽
            # ----------------------------------------------------
            if 'OFDM' in mod_type:

                bw = np.random.uniform(
                    0.5e6,
                    1.5e6
                )

            else:

                bw = np.random.uniform(
                    0.1e6,
                    0.8e6
                )

            fc = self._sample_center_frequency(
                bw,
                available_freqs[i:]
            )

            duration_ratio = np.random.uniform(
                0.3,
                1.0
            )

            t_start = np.random.uniform(
                0,
                1 - duration_ratio
            )

            t_end = (
                t_start
                + duration_ratio
            )

            sig, info = self.generate_signal(
                mod_type,
                fc,
                bw,
                t_start,
                t_end
            )

            mixture += sig

            info_list.append(
                info
            )

        return (
            mixture,
            info_list
        )

    # ============================================================
    # 噪声
    # ============================================================

    def add_realistic_noise(
        self,
        signal_clean,
        target_snr_db
    ):

        signal_power = np.mean(
            np.abs(signal_clean) ** 2
        )

        noise_power = (
            signal_power
            / (
                10 ** (
                    target_snr_db
                    / 10
                )
            )
        )

        N = self.N

        noise = np.zeros(
            N,
            dtype=complex
        )

        # Gaussian 80%
        gaussian_power = (
            0.8
            * noise_power
        )

        noise += np.sqrt(
            gaussian_power / 2
        ) * (
            np.random.randn(N)
            + 1j
            * np.random.randn(N)
        )

        # Fading-like 20%
        fading_power = (
            0.2
            * noise_power
        )

        phase = np.exp(
            1j
            * np.random.uniform(
                -np.pi / 6,
                np.pi / 6,
                N
            )
        )

        noise += np.sqrt(
            fading_power / 2
        ) * (
            np.random.randn(N)
            + 1j
            * np.random.randn(N)
        ) * phase

        current_power = np.mean(
            np.abs(noise) ** 2
        )

        if current_power > 0:

            noise = (
                noise
                * np.sqrt(
                    noise_power
                    / current_power
                )
            )

        return (
            signal_clean + noise,
            noise
        )

    # ============================================================
    # STFT
    # ============================================================

    def compute_spectrogram(self, iq_data, flatten_background=False):

        f, t, Sxx = signal.spectrogram(
            iq_data,
            fs=self.fs,
            window='hann',
            nperseg=512,
            noverlap=256,
            nfft=self.nfft,
            mode='complex'
        )

        Sxx = np.fft.fftshift(
            Sxx,
            axes=0
        )

        f = np.fft.fftshift(f)

        spec_db = 20 * np.log10(
            np.abs(Sxx) + 1e-10
        )

        # v6:
        # 不再使用逐频点中位数扣除
        # 不再使用8 dB硬阈值

        return spec_db, f, t


# ================================================================
# 保存谱图
# ================================================================

def save_spectrogram_image(
    spec_db,
    filename,
    dynamic_range_db=60
):
    """
    保存原始dB时频谱。

    不做背景中位数扣除，
    不做硬阈值。

    当前使用峰值以下固定动态范围进行显示。
    """

    vmax = float(
        np.max(spec_db)
    )

    vmin = (
        vmax
        - dynamic_range_db
    )

    plt.figure(
        figsize=(6, 4),
        dpi=100
    )

    plt.imshow(
        spec_db,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )

    plt.axis('off')

    plt.tight_layout(
        pad=0
    )

    plt.savefig(
        filename,
        bbox_inches='tight',
        pad_inches=0
    )

    plt.close()


# ================================================================
# 数据集生成
# ================================================================

def generate_dataset(
    total_samples=12000,
    output_dir='signal_dataset_v5',
    random_seed=None
):

    if random_seed is not None:
        np.random.seed(
            random_seed
        )

    train_num = int(
        total_samples * 0.8
    )

    test_num = int(
        total_samples * 0.1
    )

    val_num = (
        total_samples
        - train_num
        - test_num
    )

    splits = {
        'train': {
            'path': f'{output_dir}/train',
            'paired': True,
            'num': train_num
        },
        'test': {
            'path': f'{output_dir}/test',
            'paired': True,
            'num': test_num
        },
        'val': {
            'path': f'{output_dir}/val',
            'paired': True,
            'num': val_num
        }
    }

    for s in splits.values():

        os.makedirs(
            s['path'],
            exist_ok=True
        )

        os.makedirs(
            f"{s['path']}/low_snr",
            exist_ok=True
        )

        os.makedirs(
            f"{s['path']}/clean",
            exist_ok=True
        )

    print("\n数据集配置:")
    print(
        f"  总信号组合: "
        f"{total_samples}"
    )
    print(
        f"  Train: "
        f"{splits['train']['num']} (成对)"
    )
    print(
        f"  Test: "
        f"{splits['test']['num']} (成对)"
    )
    print(
        f"  Val: "
        f"{splits['val']['num']} (成对)"
    )
    print(
        "  低SNR: "
        "随机选择 [-10, -8, -6, -4, -2] dB"
    )
    print(
        "  高SNR参考: "
        "使用纯净信号（无噪声）"
    )
    print(
        "  调制: 20种"
    )
    print(
        "  每张图: 5-20个信号"
    )
    print(
        "  频率方向: "
        "复基带正/负频率均可"
    )
    print(
        "  保存格式: "
        "PNG图像 + metadata.json"
    )

    gen = SignalGenerator()

    low_snr_levels = [
        -10,
        -8,
        -6,
        -4,
        -2
    ]

    sample_idx = 0
    metadata_all = []

    for (
        split_name,
        split_info
    ) in splits.items():

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"生成 {split_name.upper()}: "
            f"{split_info['num']}条"
        )

        print(
            f"{'=' * 60}"
        )

        save_path = (
            split_info['path']
        )

        for i in tqdm(
            range(
                split_info['num']
            ),
            desc=split_name
        ):

            mixture_clean, sig_info = (
                gen.generate_mixture()
            )

            low_snr = int(
                np.random.choice(
                    low_snr_levels
                )
            )

            mixture_low_snr, _ = (
                gen.add_realistic_noise(
                    mixture_clean,
                    low_snr
                )
            )

            (
                spec_low_snr,
                f_axis,
                t_axis
            ) = gen.compute_spectrogram(
                mixture_low_snr
            )

            low_snr_filename = (
                f"{save_path}/low_snr/"
                f"sample_{sample_idx:05d}_"
                f"SNR{low_snr}dB.png"
            )

            save_spectrogram_image(
                spec_low_snr,
                low_snr_filename
            )

            spec_clean, _, _ = (
                gen.compute_spectrogram(
                    mixture_clean
                )
            )

            clean_filename = (
                f"{save_path}/clean/"
                f"sample_{sample_idx:05d}_"
                f"clean.png"
            )

            save_spectrogram_image(
                spec_clean,
                clean_filename
            )

            metadata_all.append(
                {
                    'sample_idx': int(
                        sample_idx
                    ),
                    'pair_id': (
                        f"sample_"
                        f"{sample_idx:05d}"
                    ),
                    'split': split_name,
                    'low_snr_db': int(
                        low_snr
                    ),
                    'low_snr_filename':
                        low_snr_filename,
                    'clean_filename':
                        clean_filename,
                    'num_signals': int(
                        len(sig_info)
                    ),
                    'mods': [
                        s['mod_type']
                        for s in sig_info
                    ],
                    'signals':
                        sig_info,
                    'spectrogram_shape': [
                        int(
                            spec_low_snr.shape[0]
                        ),
                        int(
                            spec_low_snr.shape[1]
                        )
                    ],
                    'freq_axis_hz': [
                        float(
                            f_axis[0]
                        ),
                        float(
                            f_axis[-1]
                        )
                    ],
                    'time_axis_s': [
                        float(
                            t_axis[0]
                        ),
                        float(
                            t_axis[-1]
                        )
                    ]
                }
            )

            if (
                (i + 1)
                % 200
                == 0
            ):

                mods = [
                    s['mod_type']
                    for s in sig_info
                ]

                print(
                    f"\n样本{i + 1}: "
                    f"{len(sig_info)}个信号, "
                    f"低SNR={low_snr}dB, "
                    f"类型: {set(mods)}"
                )

            sample_idx += 1

    with open(
        f'{output_dir}/metadata.json',
        'w',
        encoding='utf-8'
    ) as f:

        json.dump(
            {
                'dataset_name':
                    'WBSig-style low-SNR paired wideband signal dataset',

                'description':
                    'Signal-level paired spectrogram dataset. '
                    'Classes are modulation types. '
                    'No YOLO/detection-box labels are generated.',

                'total': int(
                    total_samples
                ),

                'splits': {
                    k: int(v['num'])
                    for k, v
                    in splits.items()
                },

                'low_snr_levels':
                    low_snr_levels,

                'mod_types':
                    gen.all_mods,

                'sample_rate_hz':
                    float(gen.fs),

                'duration_s':
                    float(gen.duration),

                'num_iq_samples':
                    int(gen.N),

                'nfft':
                    int(gen.nfft),

                'frequency_range_hz': [
                    float(
                        -gen.fs / 2
                    ),
                    float(
                        gen.fs / 2
                    )
                ],

                'center_frequency_pool_hz': [
                    float(x)
                    for x
                    in gen.freq_pool
                ],

                'image_note':
                    'Saved PNGs may be rectangular; '
                    'spectrogram matrix shape is recorded per sample.',

                'total_images': sum(
                    v['num']
                    * (
                        2
                        if v['paired']
                        else 1
                    )
                    for v
                    in splits.values()
                ),

                'samples':
                    metadata_all
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\n完成! 保存至:",
        os.path.abspath(
            output_dir
        )
    )

    return output_dir


# ================================================================
# 主入口
# ================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------
    # 当前阶段不要直接运行本文件生成完整数据集。
    #
    # 等 OFDM + 谱图预处理全部验证通过后，
    # 再正式生成新的 signal_dataset_v6。
    #
    # 如确实需要测试，可先改成：
    #
    # output = generate_dataset(
    #     total_samples=5,
    #     output_dir='signal_dataset_debug_v2',
    #     random_seed=2026
    # )
    # ------------------------------------------------------------

    print(
        "当前 generate.py 处于诊断修改阶段。"
    )

    print(
        "请运行："
    )

    print(
        "MPLBACKEND=Agg "
        "python compare_spectrogram_preprocessing.py"
    )