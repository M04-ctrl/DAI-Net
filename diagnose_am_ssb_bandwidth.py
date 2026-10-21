
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal

# ==================================================
# 1. 实验参数
# ==================================================

FS = 10_000_000
DURATION = 0.010
N = int(FS * DURATION)

TARGET_BW = 500_000

# 扫描不同消息带宽
MESSAGE_BWS = [
    350_000,
    400_000,
    450_000,
    500_000,
    550_000,
    600_000,
]

# 消息最低频率：避开直流附近的 STFT 泄漏
LOW_FREQ = 50_000

# 频谱边缘使用平滑过渡
TRANSITION = 10_000

SEED = 20261021

OUT_DIR = "am_ssb_diagnosis"
os.makedirs(OUT_DIR, exist_ok=True)


# ==================================================
# 2. 构造带限实值随机消息
# ==================================================

def generate_message(length, bandwidth, seed):
    rng = np.random.default_rng(seed)

    noise = rng.standard_normal(length)

    # 构造实值消息的双边频域滤波器
    frequencies = np.fft.fftfreq(length, d=1 / FS)

    abs_f = np.abs(frequencies)
    high_freq = LOW_FREQ + bandwidth

    mask = np.zeros(length, dtype=float)

    # 完全通过的频带
    passband = (
        (abs_f >= LOW_FREQ)
        & (abs_f <= high_freq)
    )
    mask[passband] = 1.0

    # 下边缘平滑过渡
    lower_transition = (
        (abs_f >= LOW_FREQ - TRANSITION)
        & (abs_f < LOW_FREQ)
    )
    mask[lower_transition] = 0.5 * (
        1 - np.cos(
            np.pi
            * (abs_f[lower_transition]
               - (LOW_FREQ - TRANSITION))
            / TRANSITION
        )
    )

    # 上边缘平滑过渡
    upper_transition = (
        (abs_f > high_freq)
        & (abs_f <= high_freq + TRANSITION)
    )
    mask[upper_transition] = 0.5 * (
        1 + np.cos(
            np.pi
            * (abs_f[upper_transition] - high_freq)
            / TRANSITION
        )
    )

    spectrum = np.fft.fft(noise)

    message = np.fft.ifft(
        spectrum * mask
    ).real

    message -= np.mean(message)

    std = np.std(message)

    if std < 1e-12:
        raise ValueError("Message power too small")

    message /= std

    return message


# ==================================================
# 3. 使用 Hilbert 变换构造 SSB-SC
# ==================================================

def generate_ssb(message_bw):
    iq = np.zeros(N, dtype=np.complex128)

    t0 = int(0.1 * N)
    t1 = int(0.9 * N)

    length = t1 - t0

    message = generate_message(
        length,
        message_bw,
        SEED
    )

    # 解析信号：只保留正频率边带
    base = signal.hilbert(message)

    # 将单边带占用区间中心移动到基带零频
    band_center = LOW_FREQ + message_bw / 2.0

    t_local = np.arange(length) / FS

    base = base * np.exp(
        -1j * 2 * np.pi * band_center * t_local
    )

    # RMS 归一化
    rms = np.sqrt(np.mean(np.abs(base) ** 2))

    if rms > 1e-12:
        base = base / rms

    # 与正式 generate_signal 一致的线性淡入淡出
    fade_len = min(int(0.02 * length), 2000)

    if fade_len > 10:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)

        base[:fade_len] *= fade_in
        base[-fade_len:] *= fade_out

    iq[t0:t1] = base

    return iq


# ==================================================
# 4. 计算 STFT
# ==================================================

def compute_stft(iq):
    f, t, Sxx = signal.spectrogram(
        iq,
        fs=FS,
        window="hann",
        nperseg=512,
        noverlap=256,
        nfft=512,
        mode="complex",
    )

    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    return f, t, Sxx


# ==================================================
# 5. OBW99 + 单边带功率诊断
# ==================================================

def measure_metrics(f, t, Sxx):
    # 使用稳定活动时间段
    active = (t >= 0.002) & (t <= 0.008)

    power = np.mean(
        np.abs(Sxx[:, active]) ** 2,
        axis=1
    )

    total = np.sum(power)

    if total <= 0:
        raise ValueError("Zero spectrum power")

    cumulative = np.cumsum(power) / total

    f_low = np.interp(
        0.005, cumulative, f
    )
    f_high = np.interp(
        0.995, cumulative, f
    )

    obw99 = f_high - f_low

    # 排除直流 bin，分别统计正负频率功率
    positive_power = np.sum(power[f > 0])
    negative_power = np.sum(power[f < 0])

    sideband_ratio = (
        negative_power / (positive_power + 1e-20)
    )

    negative_fraction = (
        negative_power / total
    )

    suppression_db = 10 * np.log10(
        (positive_power + 1e-20)
        / (negative_power + 1e-20)
    )

    return {
        "obw99_hz": obw99,
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "negative_fraction": negative_fraction,
        "sideband_ratio": sideband_ratio,
        "suppression_db": suppression_db,
    }


# ==================================================
# 6. 扫描不同消息带宽
# ==================================================

results = []

print("\nAM_SSB-SC Bandwidth Diagnosis")
print("=" * 75)

for message_bw in MESSAGE_BWS:
    iq = generate_ssb(message_bw)

    f, t, Sxx = compute_stft(iq)

    metrics = measure_metrics(f, t, Sxx)

    obw = metrics["obw99_hz"]
    ratio = obw / TARGET_BW

    record = {
        "message_bw_hz": message_bw,
        "obw99_hz": obw,
        "ratio": ratio,
        "negative_fraction": metrics["negative_fraction"],
        "sideband_ratio": metrics["sideband_ratio"],
        "suppression_db": metrics["suppression_db"],
        "f_low_hz": metrics["f_low_hz"],
        "f_high_hz": metrics["f_high_hz"],
    }

    results.append(record)

    print(
        f"MessageBW={message_bw / 1e3:6.0f} kHz | "
        f"OBW99={obw / 1e6:.4f} MHz | "
        f"Ratio={ratio:.3f} | "
        f"Negative={metrics['negative_fraction'] * 100:.3f}% | "
        f"Suppression={metrics['suppression_db']:.1f} dB"
    )

    # 保存代表性时频谱
    if message_bw in [350_000, 500_000, 600_000]:
        spec_db = 20 * np.log10(
            np.abs(Sxx) + 1e-10
        )

        vmax = float(np.max(spec_db))

        fig, ax = plt.subplots(figsize=(10, 5))

        im = ax.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=[
                t[0] * 1e3,
                t[-1] * 1e3,
                f[0] / 1e6,
                f[-1] / 1e6,
            ],
            vmin=vmax - 60,
            vmax=vmax,
            cmap="viridis",
        )

        ax.axhline(
            0,
            color="red",
            linestyle="--",
            linewidth=0.8,
        )

        ax.set_ylim(-0.8, 0.8)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")

        ax.set_title(
            f"SSB-SC USB: MessageBW={message_bw/1e3:.0f} kHz, "
            f"OBW99={obw/1e6:.3f} MHz"
        )

        fig.colorbar(
            im,
            ax=ax,
            label="Magnitude (dB)"
        )

        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUT_DIR,
                f"ssb_bw_{message_bw // 1000}k.png"
            ),
            dpi=150,
        )

        plt.close(fig)


# ==================================================
# 7. 保存 CSV
# ==================================================

csv_path = os.path.join(
    OUT_DIR,
    "ssb_bandwidth_scan.csv"
)

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==================================================
# 8. 绘制带宽校准曲线
# ==================================================

x = np.array(
    [r["message_bw_hz"] for r in results]
) / 1e6

y = np.array(
    [r["obw99_hz"] for r in results]
) / 1e6

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    x, y,
    "o-",
    label="Measured OBW99"
)

ax.plot(
    x, x,
    "--",
    label="Ideal message bandwidth"
)

ax.axhline(
    TARGET_BW / 1e6,
    linestyle=":",
    label="Target 0.5 MHz"
)

ax.set_xlabel("Message Bandwidth (MHz)")
ax.set_ylabel("Occupied Bandwidth (MHz)")
ax.set_title("SSB-SC Message Bandwidth vs OBW99")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUT_DIR,
        "ssb_bandwidth_curve.png"
    ),
    dpi=150
)

plt.close(fig)

best = min(
    results,
    key=lambda r: abs(r["obw99_hz"] - TARGET_BW)
)

print("\n====================================")
print("SSB-SC diagnosis completed")
print("====================================")

print(
    f"Best MessageBW: "
    f"{best['message_bw_hz'] / 1e3:.0f} kHz"
)
print(
    f"Measured OBW99: "
    f"{best['obw99_hz'] / 1e6:.4f} MHz"
)
print(
    f"Negative frequency power: "
    f"{best['negative_fraction'] * 100:.3f}%"
)
print(
    f"Sideband suppression: "
    f"{best['suppression_db']:.2f} dB"
)
print(f"Output directory: {OUT_DIR}")
