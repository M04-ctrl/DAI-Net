#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DAI-Net-Signal：20类调制信号频谱生成正确性体检

目的：
1. 对20种调制方式分别生成单信号；
2. 固定 fc=0，避免中心频率干扰；
3. 固定持续时间 t=[0.1, 0.9]；
4. 给每类信号指定一个标称 bw；
5. 使用当前 generate.py 的原始 STFT；
6. 自动估计实际占用带宽；
7. 比较：
       metadata标称带宽
       vs.
       实际STFT占用带宽
8. 为每种调制生成诊断图片；
9. 输出 CSV + JSON 汇总表。

注意：
本脚本不会修改正式数据集。
"""

from pathlib import Path
import csv
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import SignalGenerator


# ============================================================
# 1. 实验配置
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    PROJECT_DIR
    / "modulation_diagnosis"
)

IMAGE_DIR = (
    OUTPUT_DIR
    / "images"
)

RANDOM_SEED = 2026

# 固定中心频率
FC_HZ = 0.0

# 固定时间范围
T_START_RATIO = 0.10
T_END_RATIO = 0.90

# 非OFDM统一测试标称带宽
DEFAULT_BW_HZ = 0.5e6

# OFDM测试带宽
OFDM_BW_HZ = 1.0e6

# 占用带宽估计：
# 使用99%累计频谱能量范围
ENERGY_PERCENT = 0.99

# STFT显示动态范围
DISPLAY_DYNAMIC_RANGE_DB = 60


# ============================================================
# 2. 获取每类测试带宽
# ============================================================

def get_test_bw(mod_type):

    if mod_type.startswith("OFDM"):
        return OFDM_BW_HZ

    return DEFAULT_BW_HZ


# ============================================================
# 3. 根据频谱估计实际占用带宽
# ============================================================

def estimate_occupied_bandwidth(
    spec_db,
    freqs,
    times,
    t_start_ratio,
    t_end_ratio,
    total_duration,
    energy_percent=0.99
):
    """
    根据单信号的STFT估计实际占用带宽。

    方法：
    1. 只取信号真实存在的时间范围；
    2. dB恢复为线性功率；
    3. 沿时间方向平均；
    4. 根据累计频谱能量估计99% occupied bandwidth。

    返回：
        f_low
        f_high
        occupied_bw
        mean_power_spectrum
    """

    t_start = (
        t_start_ratio
        * total_duration
    )

    t_end = (
        t_end_ratio
        * total_duration
    )

    # --------------------------------------------------------
    # 找到信号实际存在的时间bin
    # --------------------------------------------------------

    time_mask = (
        (times >= t_start)
        &
        (times <= t_end)
    )

    if not np.any(time_mask):
        raise RuntimeError(
            "没有找到有效时间bin，请检查时间参数。"
        )

    active_spec_db = (
        spec_db[:, time_mask]
    )

    # --------------------------------------------------------
    # dB -> 线性功率
    #
    # spec_db = 20*log10(amplitude)
    # power = amplitude^2
    #       = 10^(spec_db/10)
    # --------------------------------------------------------

    power = (
        10.0
        ** (
            active_spec_db
            / 10.0
        )
    )

    # 时间平均功率谱
    mean_power = np.mean(
        power,
        axis=1
    )

    total_power = np.sum(
        mean_power
    )

    if (
        not np.isfinite(total_power)
        or total_power <= 0
    ):
        raise RuntimeError(
            "频谱总功率异常。"
        )

    # --------------------------------------------------------
    # 累计能量
    # --------------------------------------------------------

    cumulative = np.cumsum(
        mean_power
    )

    cumulative = (
        cumulative
        / cumulative[-1]
    )

    # 99%能量：
    # 两端各丢弃0.5%
    tail = (
        1.0
        - energy_percent
    ) / 2.0

    low_idx = np.searchsorted(
        cumulative,
        tail
    )

    high_idx = np.searchsorted(
        cumulative,
        1.0 - tail
    )

    low_idx = int(
        np.clip(
            low_idx,
            0,
            len(freqs) - 1
        )
    )

    high_idx = int(
        np.clip(
            high_idx,
            0,
            len(freqs) - 1
        )
    )

    f_low = float(
        freqs[low_idx]
    )

    f_high = float(
        freqs[high_idx]
    )

    occupied_bw = (
        f_high
        - f_low
    )

    return (
        f_low,
        f_high,
        occupied_bw,
        mean_power
    )


# ============================================================
# 4. 绘制单类诊断图
# ============================================================

def plot_diagnosis(
    mod_type,
    spec_db,
    freqs,
    times,
    info,
    estimated_low,
    estimated_high,
    estimated_bw,
    output_path
):
    """
    图中显示：
    红色虚线：metadata标称频率边界
    青色虚线：实际99%能量占用频率边界
    """

    fig, ax = plt.subplots(
        figsize=(11, 6),
        constrained_layout=True
    )

    vmax = float(
        np.max(spec_db)
    )

    vmin = (
        vmax
        - DISPLAY_DYNAMIC_RANGE_DB
    )

    extent = [
        times[0] * 1000,
        times[-1] * 1000,
        freqs[0] / 1e6,
        freqs[-1] / 1e6
    ]

    image = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest"
    )

    # --------------------------------------------------------
    # metadata 标称带宽边界
    # --------------------------------------------------------

    nominal_low = (
        info["freq_low_hz"]
        / 1e6
    )

    nominal_high = (
        info["freq_high_hz"]
        / 1e6
    )

    ax.axhline(
        nominal_low,
        linestyle="--",
        linewidth=2,
        label="Metadata lower bound"
    )

    ax.axhline(
        nominal_high,
        linestyle="--",
        linewidth=2,
        label="Metadata upper bound"
    )

    # --------------------------------------------------------
    # 实际99%能量范围
    # --------------------------------------------------------

    actual_low = (
        estimated_low
        / 1e6
    )

    actual_high = (
        estimated_high
        / 1e6
    )

    ax.axhline(
        actual_low,
        linestyle=":",
        linewidth=2,
        label="99% energy lower bound"
    )

    ax.axhline(
        actual_high,
        linestyle=":",
        linewidth=2,
        label="99% energy upper bound"
    )

    # --------------------------------------------------------
    # 时间边界
    # --------------------------------------------------------

    t_start_ms = (
        info["t_start_ratio"]
        * 10.0
    )

    t_end_ms = (
        info["t_end_ratio"]
        * 10.0
    )

    ax.axvline(
        t_start_ms,
        linestyle="--",
        linewidth=1.5
    )

    ax.axvline(
        t_end_ms,
        linestyle="--",
        linewidth=1.5
    )

    nominal_bw_mhz = (
        info["bw_hz"]
        / 1e6
    )

    actual_bw_mhz = (
        estimated_bw
        / 1e6
    )

    ratio = (
        actual_bw_mhz
        / nominal_bw_mhz
        if nominal_bw_mhz > 0
        else np.nan
    )

    ax.set_title(
        f"{mod_type} | "
        f"Nominal BW={nominal_bw_mhz:.3f} MHz | "
        f"99% OBW={actual_bw_mhz:.3f} MHz | "
        f"ratio={ratio:.2f}"
    )

    ax.set_xlabel(
        "Time (ms)"
    )

    ax.set_ylabel(
        "Frequency (MHz)"
    )

    ax.legend(
        loc="upper right",
        fontsize=8
    )

    fig.colorbar(
        image,
        ax=ax,
        label="STFT magnitude (dB)"
    )

    fig.savefig(
        output_path,
        dpi=150,
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 5. 单个调制诊断
# ============================================================

def diagnose_one(
    gen,
    mod_type
):

    bw = get_test_bw(
        mod_type
    )

    signal_iq, info = (
        gen.generate_signal(
            mod_type=mod_type,
            fc=FC_HZ,
            bw=bw,
            t_start_ratio=T_START_RATIO,
            t_end_ratio=T_END_RATIO
        )
    )

    spec_db, freqs, times = (
        gen.compute_spectrogram(
            signal_iq
        )
    )

    (
        estimated_low,
        estimated_high,
        estimated_bw,
        mean_power
    ) = estimate_occupied_bandwidth(
        spec_db,
        freqs,
        times,
        T_START_RATIO,
        T_END_RATIO,
        gen.duration,
        ENERGY_PERCENT
    )

    nominal_bw = float(
        info["bw_hz"]
    )

    ratio = (
        estimated_bw
        / nominal_bw
        if nominal_bw > 0
        else np.nan
    )

    output_path = (
        IMAGE_DIR
        / f"{mod_type}.png"
    )

    plot_diagnosis(
        mod_type,
        spec_db,
        freqs,
        times,
        info,
        estimated_low,
        estimated_high,
        estimated_bw,
        output_path
    )

    return {
        "mod_type": mod_type,

        "category":
            info["category"],

        "fc_mhz":
            info["fc_hz"] / 1e6,

        "nominal_bw_mhz":
            nominal_bw / 1e6,

        "metadata_low_mhz":
            info["freq_low_hz"] / 1e6,

        "metadata_high_mhz":
            info["freq_high_hz"] / 1e6,

        "estimated_low_mhz":
            estimated_low / 1e6,

        "estimated_high_mhz":
            estimated_high / 1e6,

        "estimated_bw_mhz":
            estimated_bw / 1e6,

        "bw_ratio":
            float(ratio),

        "t_start_ratio":
            info["t_start_ratio"],

        "t_end_ratio":
            info["t_end_ratio"]
    }


# ============================================================
# 6. CSV保存
# ============================================================

def save_csv(
    results,
    path
):

    fieldnames = [
        "mod_type",
        "category",
        "fc_mhz",
        "nominal_bw_mhz",
        "metadata_low_mhz",
        "metadata_high_mhz",
        "estimated_low_mhz",
        "estimated_high_mhz",
        "estimated_bw_mhz",
        "bw_ratio",
        "t_start_ratio",
        "t_end_ratio"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# 7. 终端打印汇总
# ============================================================

def print_summary(
    results
):

    print("\n")
    print("=" * 100)

    print(
        "20类调制实际带宽诊断结果"
    )

    print("=" * 100)

    print(
        f"{'Modulation':<12}"
        f"{'Nominal(MHz)':>15}"
        f"{'Actual(MHz)':>15}"
        f"{'Ratio':>10}"
    )

    print("-" * 100)

    for item in results:

        print(
            f"{item['mod_type']:<12}"
            f"{item['nominal_bw_mhz']:>15.3f}"
            f"{item['estimated_bw_mhz']:>15.3f}"
            f"{item['bw_ratio']:>10.2f}"
        )

    print("=" * 100)

    print(
        "\n注意："
    )

    print(
        "Ratio ≈ 1 并不自动证明调制模型完全正确；"
    )

    print(
        "Ratio偏离1则说明 metadata中的bw与实际频谱占用存在明显差异。"
    )


# ============================================================
# 8. 主函数
# ============================================================

def main():

    np.random.seed(
        RANDOM_SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    gen = SignalGenerator()

    mod_types = list(
        gen.all_mods
    )

    print("=" * 70)

    print(
        "DAI-Net-Signal：20类调制生成正确性体检"
    )

    print("=" * 70)

    print(
        "调制类别数:",
        len(mod_types)
    )

    print(
        "fc = 0 MHz"
    )

    print(
        "时间范围 = [0.10, 0.90]"
    )

    print(
        "非OFDM标称bw = 0.5 MHz"
    )

    print(
        "OFDM标称bw = 1.0 MHz"
    )

    print()

    results = []

    for index, mod_type in enumerate(
        mod_types,
        1
    ):

        print(
            f"[{index:02d}/{len(mod_types)}] "
            f"正在诊断 {mod_type} ..."
        )

        try:

            result = diagnose_one(
                gen,
                mod_type
            )

            results.append(
                result
            )

            print(
                f"    nominal="
                f"{result['nominal_bw_mhz']:.3f} MHz, "
                f"actual="
                f"{result['estimated_bw_mhz']:.3f} MHz, "
                f"ratio="
                f"{result['bw_ratio']:.2f}"
            )

        except Exception as exc:

            print(
                f"    ERROR: {exc}"
            )

    # --------------------------------------------------------
    # 保存JSON
    # --------------------------------------------------------

    json_path = (
        OUTPUT_DIR
        / "diagnosis_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 保存CSV
    # --------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "diagnosis_results.csv"
    )

    save_csv(
        results,
        csv_path
    )

    print_summary(
        results
    )

    print("\n诊断完成。")

    print(
        "结果目录:",
        OUTPUT_DIR
    )

    print(
        "图片目录:",
        IMAGE_DIR
    )

    print(
        "CSV:",
        csv_path
    )

    print(
        "JSON:",
        json_path
    )


if __name__ == "__main__":
    main()