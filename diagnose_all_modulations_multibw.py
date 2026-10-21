#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""20种调制的多带宽、多随机种子诊断（不生成正式数据集）。

直接复用 diagnose_all_modulations.py 的 OBW99 估计器，确保与原先
单点诊断采用相同的 STFT 时段、dB->功率及离散频点分位数定义。
"""
from pathlib import Path
import csv
import json
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate import SignalGenerator
from diagnose_all_modulations import estimate_occupied_bandwidth

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "modulation_multibw_diagnosis"
SEEDS = (202621, 202622, 202623)
BWS_NORMAL = (0.2e6, 0.5e6, 0.8e6)
BWS_OFDM = (0.5e6, 1.0e6, 1.5e6)
FC_HZ = 0.0
START_RATIO, END_RATIO = 0.10, 0.90
ENERGY_PERCENT = 0.99
# 阶段性宽度阈值，仅用于问题筛查，不等同物理正确性判断。
REL_TOL = 0.10


def run_one(gen, mod_type, bw, seed):
    # 此生成器当前使用 np.random 全局随机数；每次测试独立复位。
    np.random.seed(seed)
    iq, info = gen.generate_signal(
        mod_type=mod_type,
        fc=FC_HZ,
        bw=float(bw),
        t_start_ratio=START_RATIO,
        t_end_ratio=END_RATIO,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input data is complex")
        spec_db, freqs, times = gen.compute_spectrogram(iq)

    low, high, obw, power = estimate_occupied_bandwidth(
        spec_db, freqs, times,
        START_RATIO, END_RATIO,
        gen.duration, ENERGY_PERCENT,
    )
    # 该中心仅为99% OBW区间的中点；不能等同功率质心。
    center = (low + high) / 2
    nominal_center = float(info["fc_hz"])
    total_power = float(np.sum(power))
    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("invalid total spectrum power")
    if not np.isfinite(obw) or obw <= 0:
        raise ValueError("invalid occupied bandwidth")
    return {
        "mod_type": mod_type,
        "category": info["category"],
        "seed": int(seed),
        "nominal_bw_hz": float(info["bw_hz"]),
        "obw99_hz": float(obw),
        "ratio": float(obw / bw),
        "estimated_low_hz": float(low),
        "estimated_high_hz": float(high),
        "estimated_center_hz": float(center),
        "nominal_center_hz": nominal_center,
        "center_offset_hz": float(center - nominal_center),
        "center_offset_over_bw": float((center - nominal_center) / bw),
        "nominal_low_hz": float(info["freq_low_hz"]),
        "nominal_high_hz": float(info["freq_high_hz"]),
    }


def write_csv(path, rows, fields=None):
    if not rows:
        return
    columns = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def summarize(rows, mods):
    summary = []
    for mod in mods:
        bands = BWS_OFDM if mod.startswith("OFDM") else BWS_NORMAL
        for bw in bands:
            group = [r for r in rows if r["mod_type"] == mod
                     and r["nominal_bw_hz"] == bw]
            if not group:
                continue
            ratios = np.array([r["ratio"] for r in group])
            centers = np.array([r["center_offset_hz"] for r in group])
            obws = np.array([r["obw99_hz"] for r in group])
            summary.append({
                "mod_type": mod,
                "nominal_bw_mhz": bw / 1e6,
                "n": len(group),
                "obw99_mean_mhz": float(obws.mean() / 1e6),
                "obw99_std_mhz": float(obws.std(ddof=1) / 1e6) if len(group) > 1 else 0.0,
                "ratio_mean": float(ratios.mean()),
                "ratio_std": float(ratios.std(ddof=1)) if len(group) > 1 else 0.0,
                "ratio_min": float(ratios.min()),
                "ratio_max": float(ratios.max()),
                "center_offset_mean_khz": float(centers.mean() / 1e3),
                "center_offset_max_abs_khz": float(np.max(np.abs(centers)) / 1e3),
                "width_flag": "CHECK" if np.any(np.abs(ratios - 1) > REL_TOL) else "OK",
            })
    return summary


def plot_ratios(summary):
    # 每类单独小图，避免20条曲线挤在一张图上。
    fig, axes = plt.subplots(5, 4, figsize=(15, 15), sharey=False)
    for ax, mod in zip(axes.flat, list(dict.fromkeys(r["mod_type"] for r in summary))):
        group = [r for r in summary if r["mod_type"] == mod]
        xs = np.array([r["nominal_bw_mhz"] for r in group])
        ys = np.array([r["ratio_mean"] for r in group])
        errs = np.array([r["ratio_std"] for r in group])
        ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3)
        ax.axhline(1, linestyle="--", linewidth=0.9, color="gray")
        ax.axhline(1 - REL_TOL, linestyle=":", linewidth=0.7, color="gray")
        ax.axhline(1 + REL_TOL, linestyle=":", linewidth=0.7, color="gray")
        ax.set_title(mod, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Nominal BW (MHz)", fontsize=8)
        ax.set_ylabel("OBW99 / nominal BW", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "multibw_ratio_overview.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = SignalGenerator()
    mods = list(gen.all_mods)
    if len(mods) != 20:
        raise RuntimeError(f"expected 20 modulations, got {len(mods)}")

    total = sum((len(BWS_OFDM) if m.startswith("OFDM") else len(BWS_NORMAL))
                * len(SEEDS) for m in mods)
    print(f"Multiband diagnostic: {len(mods)} modulations, {total} trials")
    print("OBW estimator: imported from diagnose_all_modulations.py")
    rows, errors = [], []
    step = 0
    for mod in mods:
        bws = BWS_OFDM if mod.startswith("OFDM") else BWS_NORMAL
        for bw in bws:
            for seed in SEEDS:
                step += 1
                try:
                    r = run_one(gen, mod, bw, seed)
                    rows.append(r)
                    print(f"[{step:03d}/{total}] {mod:<9} bw={bw/1e6:.2f} "
                          f"seed={seed} ratio={r['ratio']:.3f} "
                          f"center_offset={r['center_offset_hz']/1e3:+.1f} kHz")
                except Exception as exc:
                    errors.append({"mod_type": mod, "bw_hz": float(bw),
                                   "seed": seed, "error": repr(exc)})
                    print(f"[{step:03d}/{total}] ERROR {mod}, bw={bw}, "
                          f"seed={seed}: {exc}")
    summary = summarize(rows, mods)
    write_csv(OUT / "multibw_raw.csv", rows)
    write_csv(OUT / "multibw_summary.csv", summary)
    if summary:
        plot_ratios(summary)
    with (OUT / "multibw_errors.json").open("w", encoding="utf-8") as f:
        json.dump(errors, f, ensure_ascii=False, indent=2)

    print("\n===== Summary (mean ratio +/- std) =====")
    for r in summary:
        print(f"{r['mod_type']:<9} BW={r['nominal_bw_mhz']:.2f}MHz "
              f"ratio={r['ratio_mean']:.3f}+/-{r['ratio_std']:.3f} "
              f"center={r['center_offset_mean_khz']:+.1f}kHz "
              f"{r['width_flag']}")
    print(f"\nCompleted {len(rows)}/{total} trials; errors={len(errors)}")
    print(f"Results: {OUT}")
    if errors:
        raise SystemExit("Some trials failed; see multibw_errors.json")


if __name__ == "__main__":
    main()
