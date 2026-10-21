
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from generate import SignalGenerator

FS = 10_000_000
BW = 1_000_000
N = 80_000
SEED = 20261021

OUT_DIR = "ofdm_cp_diagnosis"
os.makedirs(OUT_DIR, exist_ok=True)

gen = SignalGenerator(
    fs=FS,
    duration=0.010,
    nfft=512
)


def generate_no_cp(nfft, length, bw):
    """复现生成器的子载波映射，但不添加 CP。"""

    spacing = FS / nfft

    active = max(
        2,
        min(round(bw / spacing), nfft - 2)
    )

    const = np.array([
        1 + 0j,
        0 + 1j,
        -1 + 0j,
        0 - 1j
    ])

    blocks = []

    num_symbols = length // nfft + 1

    for _ in range(num_symbols):
        X = np.zeros(nfft, dtype=complex)

        start = nfft // 2 - active // 2

        data_idx = np.random.randint(
            0, 4, active
        )

        X[start:start + active] = const[data_idx]

        x = np.fft.ifft(np.fft.ifftshift(X))

        rms = np.sqrt(np.mean(np.abs(x) ** 2))

        if rms > 1e-12:
            x = x / rms

        blocks.append(x)

    return np.concatenate(blocks)[:length]


def measure_psd(iq):
    f, p = signal.welch(
        iq,
        fs=FS,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        nfft=8192,
        return_onesided=False,
        scaling="density"
    )

    order = np.argsort(f)

    return f[order], np.real(p[order])


def calculate_obw(f, p, fraction):
    cdf = np.cumsum(p) / np.sum(p)

    tail = (1 - fraction) / 2

    low = np.interp(tail, cdf, f)
    high = np.interp(1 - tail, cdf, f)

    return high - low


def calculate_oob(f, p):
    outside = np.abs(f) > BW / 2

    return np.sum(p[outside]) / np.sum(p)


results = []
curves = {}

for nfft in [64, 256]:

    for mode in ["with_cp", "without_cp"]:

        np.random.seed(SEED)

        if mode == "with_cp":
            iq = gen._ofdm_signal(
                nfft=nfft,
                N=N,
                bw=BW
            )
        else:
            iq = generate_no_cp(nfft, N, BW)

        f, p = measure_psd(iq)

        obw90 = calculate_obw(f, p, 0.90)
        obw95 = calculate_obw(f, p, 0.95)
        obw99 = calculate_obw(f, p, 0.99)
        oob = calculate_oob(f, p)

        results.append({
            "nfft": nfft,
            "mode": mode,
            "obw90_mhz": obw90 / 1e6,
            "obw95_mhz": obw95 / 1e6,
            "obw99_mhz": obw99 / 1e6,
            "oob_percent": oob * 100
        })

        curves[(nfft, mode)] = (f, p)

        print(
            f"OFDM_{nfft:3d} | {mode:10s} | "
            f"OBW90={obw90/1e6:.4f} | "
            f"OBW95={obw95/1e6:.4f} | "
            f"OBW99={obw99/1e6:.4f} | "
            f"OOB={oob*100:.3f}%"
        )


# 保存 CSV
with open(
    os.path.join(OUT_DIR, "cp_comparison.csv"),
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# 绘制 PSD 对比
fig, ax = plt.subplots(figsize=(11, 6))

for nfft in [64, 256]:
    for mode in ["with_cp", "without_cp"]:

        f, p = curves[(nfft, mode)]

        p_db = 10 * np.log10(
            p / np.max(p) + 1e-15
        )

        ax.plot(
            f / 1e6,
            p_db,
            linewidth=1,
            label=f"OFDM_{nfft} {mode}"
        )

ax.axvline(-0.5, color="gray", linestyle="--")
ax.axvline(0.5, color="gray", linestyle="--")

ax.set_xlim(-3, 3)
ax.set_ylim(-100, 5)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Relative PSD (dB)")
ax.set_title("OFDM CP Ablation: Spectral Leakage")

ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(OUT_DIR, "cp_psd_comparison.png"),
    dpi=160
)

plt.close(fig)

print("\nCP ablation completed.")
print("Output directory:", OUT_DIR)
