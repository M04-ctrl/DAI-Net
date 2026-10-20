
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GMSK 符号率与99%占用带宽诊断实验

只读取 generate.py 的 SignalGenerator。
不修改 generate.py，不修改原有数据集。

输出：
1. 每次实验的CSV
2. 各符号率统计汇总CSV
3. 符号率-占用带宽关系曲线
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import SignalGenerator


# ========== 1. 实验配置 ==========

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "gmsk_diagnosis"

SYMBOL_RATES = [
    0.10e6,
    0.15e6,
    0.20e6,
    0.25e6,
    0.30e6,
    0.35e6,
    0.40e6,
    0.45e6,
    0.50e6,
    0.60e6,
    0.70e6,
    0.80e6,
    0.90e6,
    1.00e6,
]

SEEDS = [2026, 2027, 2028, 2029, 2030]

ENERGY_PERCENT = 0.99
START_RATIO = 0.10
END_RATIO = 0.90


# ========== 2. 估计占用带宽 ==========

def estimate_obw(spec_db, freqs, times, duration):
    """
    采用两侧各0.5%能量的等尾99% OBW定义。
    """

    t0 = START_RATIO * duration
    t1 = END_RATIO * duration

    # 排除开始和结束附近的瞬态
    margin = 0.05 * (t1 - t0)

    mask = (
        (times >= t0 + margin)
        & (times <= t1 - margin)
    )

    if not np.any(mask):
        raise ValueError("没有有效STFT时间帧")

    # spec_db = 20 log10(|Sxx|)
    # 因此线性功率 = 10^(spec_db/10)
    power = 10.0 ** (spec_db[:, mask] / 10.0)

    mean_psd = np.mean(power, axis=1)

    if np.sum(mean_psd) <= 0:
        raise ValueError("频谱功率无效")

    cumulative = np.cumsum(mean_psd)
    cumulative /= cumulative[-1]

    tail = (1.0 - ENERGY_PERCENT) / 2.0

    low_idx = np.searchsorted(cumulative, tail)
    high_idx = np.searchsorted(cumulative, 1.0 - tail)

    low_idx = min(low_idx, len(freqs) - 1)
    high_idx = min(high_idx, len(freqs) - 1)

    return float(freqs[high_idx] - freqs[low_idx])


# ========== 3. 单次实验 ==========

def run_one(gen, target_rs, seed):
    np.random.seed(seed)

    # 目前generate.py中：
    # GMSK的symbol_rate = bw / 2
    #
    # 使用bw=2*Rs控制符号率
    input_bw = 2.0 * target_rs

    iq, info = gen.generate_signal(
        mod_type="GMSK",
        fc=0.0,
        bw=input_bw,
        t_start_ratio=START_RATIO,
        t_end_ratio=END_RATIO
    )

    # 由当前代码的整数sps获得实际符号率
    sps = max(
        int(gen.fs / info["symbol_rate"]),
        4
    )
    actual_rs = gen.fs / sps

    spec_db, freqs, times = gen.compute_spectrogram(
        iq,
        flatten_background=False
    )

    obw = estimate_obw(
        spec_db,
        freqs,
        times,
        gen.duration
    )

    return {
        "seed": seed,
        "target_rs_mhz": target_rs / 1e6,
        "actual_rs_mhz": actual_rs / 1e6,
        "input_bw_mhz": input_bw / 1e6,
        "obw99_mhz": obw / 1e6,
        "obw_to_rs": obw / actual_rs
    }


# ========== 4. 保存CSV ==========

def save_csv(path, rows):
    with open(
        path, "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


# ========== 5. 主实验 ==========

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gen = SignalGenerator()

    all_rows = []
    summary = []

    print("=" * 75)
    print("GMSK: Symbol Rate vs 99% Occupied Bandwidth")
    print("=" * 75)

    for target_rs in SYMBOL_RATES:

        group = []

        for seed in SEEDS:
            result = run_one(gen, target_rs, seed)
            group.append(result)
            all_rows.append(result)

        rs_values = np.array([
            r["actual_rs_mhz"] for r in group
        ])

        obw_values = np.array([
            r["obw99_mhz"] for r in group
        ])

        rs_mean = float(np.mean(rs_values))
        obw_mean = float(np.mean(obw_values))
        obw_std = float(np.std(obw_values, ddof=1))

        summary.append({
            "target_rs_mhz": target_rs / 1e6,
            "actual_rs_mhz": rs_mean,
            "mean_obw99_mhz": obw_mean,
            "std_obw99_mhz": obw_std,
            "obw_to_rs": obw_mean / rs_mean
        })

        print(
            f"Target Rs={target_rs/1e6:.3f} Msyms/s | "
            f"Actual Rs={rs_mean:.3f} | "
            f"OBW99={obw_mean:.3f} ± {obw_std:.3f} MHz | "
            f"OBW/Rs={obw_mean/rs_mean:.3f}"
        )

    # ========== 6. 线性拟合 ==========

    x = np.array([
        r["actual_rs_mhz"] for r in summary
    ])

    y = np.array([
        r["mean_obw99_mhz"] for r in summary
    ])

    # 普通一阶线性回归：OBW = a*Rs + b
    a, b = np.polyfit(x, y, 1)
    y_fit = a * x + b

    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0 else float("nan")
    )

    print("\n拟合结果：")
    print(f"OBW99(MHz) = {a:.4f} * Rs(Msym/s) + {b:.4f}")
    print(f"R² = {r2:.4f}")

    # ========== 7. 生成关系曲线 ==========

    fig, ax = plt.subplots(
        figsize=(9, 6), constrained_layout=True
    )

    yerr = [
        r["std_obw99_mhz"] for r in summary
    ]

    ax.errorbar(
        x, y, yerr=yerr,
        fmt="o",
        capsize=4,
        label="GMSK measured 99% OBW"
    )

    x_line = np.linspace(
        min(x), max(x), 100
    )

    ax.plot(
        x_line,
        a * x_line + b,
        label=f"Linear fit (R²={r2:.3f})"
    )

    ax.set_xlabel("Actual Symbol Rate (Msymbol/s)")
    ax.set_ylabel("99% Occupied Bandwidth (MHz)")
    ax.set_title("GMSK: Symbol Rate vs Occupied Bandwidth")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / "gmsk_rs_vs_obw.png",
        dpi=150
    )

    plt.close(fig)

    # ========== 8. 保存数据 ==========

    save_csv(
        OUTPUT_DIR / "gmsk_all_trials.csv",
        all_rows
    )

    save_csv(
        OUTPUT_DIR / "gmsk_summary.csv",
        summary
    )

    print("\n实验完成！")
    print("输出目录:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
