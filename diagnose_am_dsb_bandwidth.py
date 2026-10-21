
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

# 与正式生成器一致的基本配置
FS = 10_000_000
DURATION = 0.010
N = int(FS * DURATION)

TARGET_BW = 500_000
MOD_INDEX = 0.8

# 从低消息频率扫描到目标理论值附近
MOD_FREQS = [
    205_000,
    210_000,
    215_000,
    220_000,
    221_000,
    225_000,
    230_000,
]

OUT_DIR = "am_dsb_diagnosis_fine"
os.makedirs(OUT_DIR, exist_ok=True)


def generate_am_dsb(mod_f):
    iq = np.zeros(N, dtype=np.complex128)
    start = int(0.1 * N)
    end = int(0.9 * N)
    length = end - start

    t = np.arange(length) / FS

    base = 1.0 + MOD_INDEX * np.sin(
        2 * np.pi * mod_f * t
    )

    # 与正式 generate_signal() 相同的淡入淡出
    fade_len = min(int(0.02 * length), 2000)
    envelope = np.ones(length)

    if fade_len > 10:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        envelope[:fade_len] *= fade_in
        envelope[-fade_len:] *= fade_out

    iq[start:end] = base * envelope
    return iq


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


def measure_obw99(f, t, Sxx):
    # 排除信号起止门控区域
    active = (t >= 0.002) & (t <= 0.008)

    power = np.mean(
        np.abs(Sxx[:, active]) ** 2,
        axis=1,
    )

    total = power.sum()

    if total <= 0:
        raise ValueError("Zero spectrum power")

    cumulative = np.cumsum(power) / total

    low = np.interp(0.005, cumulative, f)
    high = np.interp(0.995, cumulative, f)

    return high - low


results = []

print("\nAM_DSB Frequency Bandwidth Diagnosis")
print("=" * 70)

for mod_f in MOD_FREQS:
    iq = generate_am_dsb(mod_f)
    f, t, Sxx = compute_stft(iq)

    obw = measure_obw99(f, t, Sxx)
    theoretical_bw = 2 * mod_f
    ratio = obw / TARGET_BW

    results.append({
        "mod_f_hz": mod_f,
        "theoretical_bw_hz": theoretical_bw,
        "obw99_hz": obw,
        "ratio": ratio,
    })

    print(
        f"fm={mod_f / 1e3:7.1f} kHz | "
        f"BW_theory={theoretical_bw / 1e6:.3f} MHz | "
        f"OBW99={obw / 1e6:.4f} MHz | "
        f"Ratio={ratio:.3f}"
    )

    if mod_f in [5_000, 150_000, 250_000]:
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
                f[-1] / 1e6
            ],
            vmin=vmax - 60,
            vmax=vmax,
            cmap="viridis"
        )

        ax.set_ylim(-0.7, 0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (MHz)")
        ax.set_title(
            f"AM_DSB fm={mod_f / 1e3:.0f} kHz, "
            f"OBW99={obw / 1e6:.3f} MHz"
        )
        fig.colorbar(im, ax=ax, label="Magnitude (dB)")
        fig.tight_layout()

        fig.savefig(
            os.path.join(
                OUT_DIR,
                f"am_dsb_fm_{mod_f // 1000}k.png"
            ),
            dpi=150
        )
        plt.close(fig)


with open(
    os.path.join(OUT_DIR, "am_dsb_frequency_scan.csv"),
    "w",
    newline=""
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "mod_f_hz",
            "theoretical_bw_hz",
            "obw99_hz",
            "ratio"
        ]
    )
    writer.writeheader()
    writer.writerows(results)


frequencies = np.array(
    [r["mod_f_hz"] for r in results]
) / 1e3

measured = np.array(
    [r["obw99_hz"] for r in results]
) / 1e6

theoretical = np.array(
    [r["theoretical_bw_hz"] for r in results]
) / 1e6

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    frequencies, measured,
    "o-", label="Measured 99% OBW"
)
ax.plot(
    frequencies, theoretical,
    "--", label="Theoretical 2fm"
)
ax.axhline(
    TARGET_BW / 1e6,
    linestyle=":",
    label="Target 0.5 MHz"
)
ax.set_xlabel("Message Frequency (kHz)")
ax.set_ylabel("Bandwidth (MHz)")
ax.set_title("AM_DSB Message Frequency vs OBW99")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(
    os.path.join(OUT_DIR, "am_dsb_obw_curve.png"),
    dpi=150
)
plt.close(fig)

best = min(
    results,
    key=lambda r: abs(r["obw99_hz"] - TARGET_BW)
)

print("\nDiagnosis completed")
print(f"Best scanned fm = {best['mod_f_hz']/1e3:.1f} kHz")
print(f"Measured OBW99 = {best['obw99_hz']/1e6:.4f} MHz")
print(f"Output directory = {OUT_DIR}")
