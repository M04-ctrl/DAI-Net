
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
BW = 1_000_000
N = 80_000
SEED = 20261021

OUT_DIR = "ofdm_wola_diagnosis"
os.makedirs(OUT_DIR, exist_ok=True)

CONFIGS = {
    64: [0, 4, 8],
    256: [0, 16, 32],
}


# ==========================================
# 2. 生成同一组 OFDM 频域数据
# ==========================================

def generate_symbols(nfft, length, bw, seed):
    rng = np.random.default_rng(seed)

    cp_len = nfft // 4
    symbol_len = nfft + cp_len
    n_symbols = length // symbol_len + 2

    delta_f = FS / nfft
    n_active = int(round(bw / delta_f))
    n_active = max(2, min(n_active, nfft - 2))

    const = np.array([
        1 + 0j,
        0 + 1j,
        -1 + 0j,
        0 - 1j
    ])

    start = nfft // 2 - n_active // 2

    frequency_symbols = []
    time_symbols = []

    for _ in range(n_symbols):
        X_shifted = np.zeros(nfft, dtype=complex)

        idx = rng.integers(0, 4, n_active)
        X_shifted[start:start + n_active] = const[idx]

        x = np.fft.ifft(np.fft.ifftshift(X_shifted))

        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        if rms > 1e-12:
            x = x / rms
            X_shifted = X_shifted / rms

        frequency_symbols.append(X_shifted)
        time_symbols.append(x)

    return (
        np.array(time_symbols),
        np.array(frequency_symbols),
        cp_len,
        n_active
    )


# ==========================================
# 3. 原始 CP-OFDM
# ==========================================

def build_original(time_symbols, cp_len):
    blocks = []

    for x in time_symbols:
        cp = x[-cp_len:]
        blocks.append(np.concatenate([cp, x]))

    return np.concatenate(blocks)


# ==========================================
# 4. WOLA：短循环后缀 + 互补窗 + 重叠相加
# ==========================================

def build_wola(time_symbols, cp_len, overlap):
    if overlap == 0:
        return build_original(time_symbols, cp_len)

    if overlap > cp_len:
        raise ValueError("Overlap must not exceed CP length")

    nfft = time_symbols.shape[1]
    symbol_len = nfft + cp_len

    total_len = (
        len(time_symbols) * symbol_len + overlap
    )

    output = np.zeros(total_len, dtype=complex)

    # 相邻符号的重叠区采用互补余弦窗
    u = (np.arange(overlap) + 0.5) / overlap

    fade_in = np.sin(np.pi * u / 2) ** 2
    fade_out = 1.0 - fade_in

    for k, x in enumerate(time_symbols):
        cp = x[-cp_len:]

        # 循环后缀
        suffix = x[:overlap]

        block = np.concatenate([
            cp, x, suffix
        ]).astype(complex)

        block[:overlap] *= fade_in
        block[-overlap:] *= fade_out

        start = k * symbol_len
        end = start + len(block)

        output[start:end] += block

    return output


# ==========================================
# 5. Welch PSD
# ==========================================

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


# ==========================================
# 6. OBW 与带外功率
# ==========================================

def calculate_obw(f, p, fraction):
    power = np.maximum(p, 0)
    cdf = np.cumsum(power) / np.sum(power)

    tail = (1.0 - fraction) / 2.0

    low = np.interp(tail, cdf, f)
    high = np.interp(1.0 - tail, cdf, f)

    return high - low


def calculate_oob(f, p):
    outside = np.abs(f) > BW / 2

    return np.sum(p[outside]) / np.sum(p)


# ==========================================
# 7. 有效 FFT 区间的解调 EVM
# ==========================================

def calculate_evm(iq, reference, nfft, cp_len):
    symbol_len = nfft + cp_len

    # 跳过首尾，避免观测区间截断影响
    n_available = (
        len(iq) - cp_len - nfft
    ) // symbol_len + 1

    n_check = min(
        len(reference) - 2,
        n_available - 1
    )

    if n_check <= 0:
        raise ValueError("Not enough OFDM symbols")

    errors = []
    powers = []

    for k in range(1, n_check + 1):
        start = k * symbol_len + cp_len
        segment = iq[start:start + nfft]

        Y = np.fft.fftshift(np.fft.fft(segment))
        X = reference[k]

        active = np.abs(X) > 1e-12

        errors.append(
            np.sum(np.abs(Y[active] - X[active]) ** 2)
        )

        powers.append(
            np.sum(np.abs(X[active]) ** 2)
        )

    return np.sqrt(
        np.sum(errors) / np.sum(powers)
    ) * 100


# ==========================================
# 8. 控制变量实验
# ==========================================

results = []
curves = {}

for nfft, overlaps in CONFIGS.items():

    (
        time_symbols,
        reference,
        cp_len,
        n_active
    ) = generate_symbols(
        nfft, N, BW, SEED
    )

    print(f"\n===== OFDM_{nfft} =====")
    print(f"CP length: {cp_len}")
    print(f"Active subcarriers: {n_active}")

    for overlap in overlaps:
        if overlap == 0:
            mode = "original"
            iq = build_original(
                time_symbols, cp_len
            )
        else:
            mode = f"WOLA_{overlap}"
            iq = build_wola(
                time_symbols, cp_len, overlap
            )

        # 固定观测长度，确保对比公平
        iq = iq[:N]

        f, p = measure_psd(iq)

        obw90 = calculate_obw(f, p, 0.90)
        obw95 = calculate_obw(f, p, 0.95)
        obw99 = calculate_obw(f, p, 0.99)
        oob = calculate_oob(f, p)

        evm = calculate_evm(
            iq, reference, nfft, cp_len
        )

        record = {
            "modulation": f"OFDM_{nfft}",
            "mode": mode,
            "overlap_samples": overlap,
            "obw90_mhz": obw90 / 1e6,
            "obw95_mhz": obw95 / 1e6,
            "obw99_mhz": obw99 / 1e6,
            "oob_percent": oob * 100,
            "evm_percent": evm
        }

        results.append(record)
        curves[(nfft, mode)] = (f, p)

        print(
            f"{mode:10s} | "
            f"OBW90={obw90/1e6:.4f} | "
            f"OBW95={obw95/1e6:.4f} | "
            f"OBW99={obw99/1e6:.4f} | "
            f"OOB={oob*100:.3f}% | "
            f"EVM={evm:.6f}%"
        )


# ==========================================
# 9. 保存 CSV
# ==========================================

with open(
    os.path.join(OUT_DIR, "wola_metrics.csv"),
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==========================================
# 10. 分别绘制两类 OFDM PSD
# ==========================================

for nfft, overlaps in CONFIGS.items():

    fig, ax = plt.subplots(figsize=(11, 6))

    for overlap in overlaps:
        mode = (
            "original" if overlap == 0
            else f"WOLA_{overlap}"
        )

        f, p = curves[(nfft, mode)]

        p_db = 10 * np.log10(
            p / np.max(p) + 1e-15
        )

        ax.plot(
            f / 1e6,
            p_db,
            linewidth=1,
            label=mode
        )

    ax.axvline(
        -BW / 2e6,
        color="gray",
        linestyle="--"
    )
    ax.axvline(
        BW / 2e6,
        color="gray",
        linestyle="--"
    )

    ax.set_xlim(-3, 3)
    ax.set_ylim(-100, 5)

    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Relative PSD (dB)")
    ax.set_title(
        f"OFDM_{nfft}: Original vs WOLA"
    )

    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            OUT_DIR,
            f"ofdm_{nfft}_wola_psd.png"
        ),
        dpi=160
    )

    plt.close(fig)


print("\nWOLA diagnosis completed.")
print("Output directory:", OUT_DIR)
