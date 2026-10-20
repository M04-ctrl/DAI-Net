
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAI-Net-Signal 频谱预处理诊断实验 v1.0

目标：
验证逐频点中位数扣除和8 dB硬阈值，
是否导致长持续时间信号的时频结构丢失。

不修改正式数据集，也不覆盖原始图片。
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import SignalGenerator


# ========== 1. 实验配置 ==========

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "spectrogram_comparison"

RANDOM_SEED = 2026
THRESHOLD_DB = 8.0

# 所有处理后相对谱图统一使用这个显示范围
DISPLAY_MIN = -5.0
DISPLAY_MAX = 30.0


# ========== 2. 构建四种频谱 ==========

def compute_four_spectrograms(gen, iq_data):

    # 调用原有STFT函数，但关闭背景处理
    raw_db, freqs, times = gen.compute_spectrogram(
        iq_data,
        flatten_background=False
    )

    # A：全局背景参考，仅平移dB零点
    global_background = float(np.median(raw_db))
    raw_relative = raw_db - global_background

    # B：逐频点中位数扣除
    freq_median = np.median(
        raw_db, axis=1, keepdims=True
    )
    median_sub = raw_db - freq_median

    # C：复现原generate.py的处理逻辑
    median_threshold = np.where(
        median_sub < THRESHOLD_DB,
        0.0,
        median_sub
    )

    # D：全局背景扣除（作为候选替代方式）
    global_sub = raw_db - global_background

    results = {
        "A_raw_relative": raw_relative,
        "B_median_sub": median_sub,
        "C_median_threshold": median_threshold,
        "D_global_sub": global_sub,
    }

    return results, freqs, times, raw_db


# ========== 3. 可视化对比 ==========

def plot_comparison(
    results,
    freqs,
    times,
    output_path,
    title
):

    fig, axes = plt.subplots(
        2, 2,
        figsize=(14, 9),
        constrained_layout=True
    )

    names = [
        ("A_raw_relative",
         "A: Raw STFT (global reference)"),
        ("B_median_sub",
         "B: Per-frequency median subtraction"),
        ("C_median_threshold",
         "C: Median subtraction + 8 dB threshold"),
        ("D_global_sub",
         "D: Global background subtraction"),
    ]

    extent = [
        times[0] * 1000,
        times[-1] * 1000,
        freqs[0] / 1e6,
        freqs[-1] / 1e6
    ]

    for ax, (key, label) in zip(
        axes.flat, names
    ):
        image = ax.imshow(
            results[key],
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=DISPLAY_MIN,
            vmax=DISPLAY_MAX,
            interpolation="nearest"
        )

        ax.set_title(label)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

    fig.colorbar(
        image,
        ax=axes,
        label="Relative level (dB)",
        shrink=0.75
    )

    fig.suptitle(title, fontsize=15)
    fig.savefig(
        output_path,
        dpi=150,
        facecolor="white"
    )
    plt.close(fig)


# ========== 4. 数值诊断 ==========

def print_statistics(name, results):

    print(f"\n===== {name} =====")

    for key, spec in results.items():
        print(
            f"{key:24s} "
            f"min={np.min(spec):8.2f}  "
            f"max={np.max(spec):8.2f}  "
            f"mean={np.mean(spec):8.2f}  "
            f"nonzero={np.mean(spec != 0):.2%}  "
            f"above8dB={np.mean(spec >= 8):.2%}"
        )


# ========== 5. 单信号场景 ==========

def run_single_signal(gen):

    print("\n生成单信号：长持续时间 QPSK")

    # 持续整张谱图85%的时间
    signal_iq, info = gen.generate_signal(
        mod_type="QPSK",
        fc=1.5e6,
        bw=0.5e6,
        t_start_ratio=0.10,
        t_end_ratio=0.95
    )

    results, freqs, times, _ = (
        compute_four_spectrograms(
            gen, signal_iq
        )
    )

    print("信号参数:", info)

    print_statistics(
        "Single QPSK",
        results
    )

    plot_comparison(
        results,
        freqs,
        times,
        OUTPUT_DIR / "single_qpsk_comparison.png",
        "Long-duration QPSK: preprocessing comparison"
    )


# ========== 6. 多信号混合场景 ==========

def run_mixture(gen):

    print("\n生成多信号混合场景")

    mixture_iq, infos = gen.generate_mixture(
        num_signals=6
    )

    print("调制类型:")
    for idx, info in enumerate(infos, 1):
        print(
            f"{idx}. {info['mod_type']:10s} "
            f"fc={info['fc_hz']/1e6:6.3f} MHz  "
            f"bw={info['bw_hz']/1e6:6.3f} MHz  "
            f"time=[{info['t_start_ratio']:.3f}, "
            f"{info['t_end_ratio']:.3f}]"
        )
    results, freqs, times, _ = (
        compute_four_spectrograms(
            gen, mixture_iq
        )
    )

    print_statistics(
        "Mixture of 6 signals",
        results
    )

    plot_comparison(
        results,
        freqs,
        times,
        OUTPUT_DIR / "mixture_comparison.png",
        "Mixed modulation signals: preprocessing comparison"
    )

def run_single_ofdm(gen):

    print("\n生成单信号：OFDM_256")

    signal_iq, info = gen.generate_signal(
        mod_type="OFDM_256",
        fc=0.0,
        bw=1.0e6,
        t_start_ratio=0.10,
        t_end_ratio=0.90
    )

    results, freqs, times, _ = compute_four_spectrograms(
        gen,
        signal_iq
    )

    print("OFDM参数:", info)

    print_statistics(
        "Single OFDM_256",
        results
    )

    plot_comparison(
        results,
        freqs,
        times,
        OUTPUT_DIR / "single_ofdm256_comparison.png",
        "OFDM-256: preprocessing comparison"
    )


# ========== 7. 主程序 ==========

def main():

    np.random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 60)
    print("DAI-Net-Signal: STFT preprocessing diagnosis")
    print("=" * 60)

    gen = SignalGenerator()

    run_single_signal(gen)
    run_single_ofdm(gen)
    run_mixture(gen)

    print("\n实验完成！")
    print("输出目录:", OUTPUT_DIR)
    print("生成图片:")
    print("1. single_qpsk_comparison.png")
    print("2. mixture_comparison.png")
    print("3. mixture_comparison.png")

if __name__ == "__main__":
    main()
