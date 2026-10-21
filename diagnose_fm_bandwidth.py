
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal

# ==========================================
# 1. 实验参数
# ==========================================

FS = 10_000_000
DURATION = 0.010
N = int(FS * DURATION)

MOD_F = 10_000
TARGET_BW = 500_000

BETA_VALUES = [21.0, 22.0, 22.46, 23.0, 24.0]

NPERSEG = 512
NOVERLAP = 256
NFFT = 512

OUTPUT_DIR = "fm_diagnosis_fine"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. 生成单音 FM 测试波形
# ==========================================

def generate_fm(beta):
    iq = np.zeros(N, dtype=np.complex128)

    t0 = int(0.1 * N)
    t1 = int(0.9 * N)
    length = t1 - t0

    t = np.arange(length) / FS

    message = np.sin(2 * np.pi * MOD_F * t)

    base = np.exp(1j * beta * message)

    # 平滑信号起止边界，减小门控泄漏
    fade_len = 100
    envelope = np.ones(length)

    ramp = np.sin(
        np.linspace(0, np.pi / 2, fade_len)
    ) ** 2

    envelope[:fade_len] = ramp
    envelope[-fade_len:] = ramp[::-1]

    iq[t0:t1] = base * envelope

    return iq


# ==========================================
# 3. STFT：与现有生成器参数一致
# ==========================================

def compute_spectrum(iq):
    f, t, Sxx = signal.spectrogram(
        iq,
        fs=FS,
        window="hann",
        nperseg=NPERSEG,
        noverlap=NOVERLAP,
        nfft=NFFT,
        mode="complex"
    )

    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    spec_db = 20 * np.log10(
        np.abs(Sxx) + 1e-10
    )

    return f, t, Sxx, spec_db


# ==========================================
# 4. 计算 99% 占用带宽
# ==========================================

def measure_obw(f, t, Sxx):
    # 只使用信号稳定活动的中间时段
    active = (t >= 0.002) & (t <= 0.008)

    power = np.mean(
        np.abs(Sxx[:, active]) ** 2,
        axis=1
    )

    power = np.maximum(power, 0)
    total = np.sum(power)

    if total <= 0:
        raise ValueError("Spectrum has zero power")

    cumulative = np.cumsum(power) / total

    # 两端各排除 0.5% 能量
    f_low = np.interp(0.005, cumulative, f)
    f_high = np.interp(0.995, cumulative, f)

    return f_high - f_low, f_low, f_high


# ==========================================
# 5. 扫描 beta 并保存结果
# ==========================================

results = []

for beta in BETA_VALUES:
    iq = generate_fm(beta)

    f, t, Sxx, spec_db = compute_spectrum(iq)

    obw, f_low, f_high = measure_obw(f, t, Sxx)

    carson_bw = 2 * (beta + 1) * MOD_F

    ratio = obw / TARGET_BW

    results.append({
        "beta": beta,
        "obw99_hz": obw,
        "carson_bw_hz": carson_bw,
        "ratio": ratio
    })

    print(
        f"beta={beta:>4} | "
        f"OBW99={obw / 1e6:.4f} MHz | "
        f"Carson={carson_bw / 1e6:.4f} MHz | "
        f"Ratio={ratio:.3f}"
    )

    # 只为有代表性的 beta 保存时频谱
    if beta in [5, 15, 25, 35]:
        fig, ax = plt.subplots(figsize=(10, 5))

        vmax = np.max(spec_db)

        im = ax.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=[
                t[0] * 1e3,
                t[-1] * 1e3,
                f[0] / 1e6,
                f[-1] / 1e6
            ],
            vmin=vmax - 60,
            vmax=vmax,
            cmap="viridis"
        )

        ax.set_ylim(-1, 1)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(
            f"Single-tone FM: beta={beta}, "
            f"OBW99={obw / 1e6:.3f} MHz"
        )

        fig.colorbar(im, ax=ax, label="Magnitude (dB)")

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"fm_beta_{beta}.png"
            ),
            dpi=150
        )
        plt.close(fig)


# ==========================================
# 6. 保存 CSV
# ==========================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "fm_beta_obw.csv"
)

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "beta",
            "obw99_hz",
            "carson_bw_hz",
            "ratio"
        ]
    )
    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 7. 绘制 beta-OBW 曲线
# ==========================================

betas = np.array(
    [r["beta"] for r in results]
)

obws = np.array(
    [r["obw99_hz"] for r in results]
) / 1e6

carsons = np.array(
    [r["carson_bw_hz"] for r in results]
) / 1e6

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    betas, obws,
    "o-",
    label="Measured 99% OBW"
)

ax.plot(
    betas, carsons,
    "--",
    label="Carson approximation"
)

ax.axhline(
    TARGET_BW / 1e6,
    linestyle=":",
    label="Target 0.5 MHz"
)

ax.set_xlabel("FM Modulation Index (beta)")
ax.set_ylabel("Bandwidth (MHz)")
ax.set_title("FM Modulation Index vs Occupied Bandwidth")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "fm_beta_obw_curve.png"
    ),
    dpi=150
)

plt.close(fig)

best = min(
    results,
    key=lambda r: abs(r["obw99_hz"] - TARGET_BW)
)

print("\n====================================")
print("FM bandwidth diagnosis completed")
print("====================================")
print(f"Best scanned beta: {best['beta']}")
print(f"Measured OBW99: {best['obw99_hz']/1e6:.4f} MHz")
print(f"Target bandwidth: {TARGET_BW/1e6:.4f} MHz")
print(f"Output directory: {OUTPUT_DIR}")
