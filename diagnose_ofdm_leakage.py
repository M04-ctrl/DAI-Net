
import os
import csv
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import signal
from generate import SignalGenerator

# ==========================================
# 1. 参数
# ==========================================

FS = 10_000_000
BW = 1_000_000
DURATION = 0.008
N = int(FS * DURATION)

OUTPUT_DIR = "ofdm_leakage_diagnosis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

gen = SignalGenerator(
    fs=FS,
    duration=0.010,
    nfft=512
)


# ==========================================
# 2. 计算能量占用带宽
# ==========================================

def calculate_obw(f, power, fraction):
    order = np.argsort(f)
    f = f[order]
    power = np.maximum(power[order], 0)

    total = np.sum(power)

    if total <= 0:
        raise ValueError("Zero spectrum power")

    cdf = np.cumsum(power) / total
    tail = (1.0 - fraction) / 2.0

    f_low = np.interp(tail, cdf, f)
    f_high = np.interp(1.0 - tail, cdf, f)

    return f_high - f_low


# ==========================================
# 3. 带外功率
# ==========================================

def calculate_oob(f, power, bw):
    total = np.sum(power)

    outside = np.abs(f) > bw / 2.0

    oob_fraction = (
        np.sum(power[outside]) / total
    )

    return oob_fraction


# ==========================================
# 4. STFT 与 Welch PSD
# ==========================================

def stft_power(iq):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Input data is complex"
        )

        f, t, Sxx = signal.spectrogram(
            iq,
            fs=FS,
            window="hann",
            nperseg=512,
            noverlap=256,
            nfft=512,
            mode="complex",
            return_onesided=False
        )

    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    power = np.mean(np.abs(Sxx) ** 2, axis=1)

    return f, power


def welch_power(iq):
    f, psd = signal.welch(
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

    return f[order], psd[order].real


# ==========================================
# 5. 分别诊断两种 OFDM
# ==========================================

results = []
plot_data = {}

for nfft in [64, 256]:
    np.random.seed(20261021)

    iq = gen._ofdm_signal(
        nfft=nfft,
        N=N,
        bw=BW
    )

    f_stft, p_stft = stft_power(iq)
    f_psd, p_psd = welch_power(iq)

    delta_f = FS / nfft
    num_active = round(BW / delta_f)

    record = {
        "modulation": f"OFDM_{nfft}",
        "active_subcarriers": num_active,
        "subcarrier_spacing_hz": delta_f
    }

    print(f"\n===== OFDM_{nfft} =====")
    print(f"Active subcarriers: {num_active}")
    print(f"Spacing: {delta_f/1e3:.3f} kHz")

    for method, f, p in [
        ("STFT512", f_stft, p_stft),
        ("Welch8192", f_psd, p_psd)
    ]:
        print(f"\n{method}")

        for fraction in [0.90, 0.95, 0.99]:
            obw = calculate_obw(f, p, fraction)

            key = f"{method}_OBW{int(fraction*100)}_MHz"
            record[key] = obw / 1e6

            print(
                f"OBW{int(fraction*100)}: "
                f"{obw/1e6:.4f} MHz"
            )

        oob = calculate_oob(f, p, BW)

        record[f"{method}_OOB_percent"] = oob * 100

        print(f"OOB power: {oob*100:.3f}%")

    results.append(record)
    plot_data[nfft] = (f_psd, p_psd)


# ==========================================
# 6. 保存比较曲线
# ==========================================

fig, ax = plt.subplots(figsize=(10, 5))

for nfft in [64, 256]:
    f, p = plot_data[nfft]

    p_db = 10 * np.log10(
        p / np.max(p) + 1e-15
    )

    ax.plot(
        f / 1e6,
        p_db,
        label=f"OFDM_{nfft}",
        linewidth=1.0
    )

ax.axvline(
    -BW / 2e6,
    linestyle="--",
    color="gray"
)

ax.axvline(
    BW / 2e6,
    linestyle="--",
    color="gray"
)

ax.set_xlim(-3, 3)
ax.set_ylim(-100, 5)

ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Relative PSD (dB)")
ax.set_title("OFDM Spectral Leakage Comparison")
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "ofdm_psd_comparison.png"
    ),
    dpi=160
)

plt.close(fig)


# ==========================================
# 7. 保存 CSV
# ==========================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "ofdm_leakage_metrics.csv"
)

with open(csv_path, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

print("\nDiagnosis completed")
print(f"Output directory: {OUTPUT_DIR}")
