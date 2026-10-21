#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OFDM WOLA stability test: paired seeds, STFT512/Welch8192, OBW/OOB/EVM."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal

FS = 10_000_000
BW = 1_000_000
N = 80_000
SEEDS = [20261021, 20261022, 20261023, 20261024, 20261025]
CONFIGS = {64: 8, 256: 32}
OUT_DIR = Path('ofdm_wola_stability')
OUT_DIR.mkdir(exist_ok=True)


def make_symbols(nfft, length, bw, seed):
    rng = np.random.default_rng(seed)
    cp_len = nfft // 4
    symbol_len = nfft + cp_len
    n_symbols = length // symbol_len + 2
    spacing = FS / nfft
    active = max(2, min(int(round(bw / spacing)), nfft - 2))
    const = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j])
    start = nfft // 2 - active // 2
    xs, refs = [], []
    for _ in range(n_symbols):
        X = np.zeros(nfft, dtype=complex)
        X[start:start + active] = const[rng.integers(0, 4, active)]
        x = np.fft.ifft(np.fft.ifftshift(X))
        rms = np.sqrt(np.mean(np.abs(x) ** 2))
        x = x / rms
        X = X / rms
        xs.append(x)
        refs.append(X)
    return np.asarray(xs), np.asarray(refs), cp_len


def make_waveform(xs, cp_len, overlap):
    nfft = xs.shape[1]
    hop = nfft + cp_len
    if not 0 <= overlap <= cp_len:
        raise ValueError('overlap must be between 0 and CP length')
    if overlap == 0:
        return np.concatenate([np.concatenate([x[-cp_len:], x]) for x in xs])
    out = np.zeros(len(xs) * hop + overlap, dtype=np.complex128)
    u = (np.arange(overlap) + 0.5) / overlap
    fade_in = np.sin(0.5 * np.pi * u) ** 2
    fade_out = 1 - fade_in
    for k, x in enumerate(xs):
        block = np.concatenate([x[-cp_len:], x, x[:overlap]]).copy()
        block[:overlap] *= fade_in
        block[-overlap:] *= fade_out
        pos = k * hop
        out[pos:pos + len(block)] += block
    return out


def estimate_spectrum(iq, method):
    if method == 'STFT512':
        f, _, spectrum = signal.spectrogram(
            iq, fs=FS, window='hann', nperseg=512, noverlap=256,
            nfft=512, mode='complex', return_onesided=False)
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
    elif method == 'Welch8192':
        f, power = signal.welch(
            iq, fs=FS, window='hann', nperseg=8192, noverlap=4096,
            nfft=8192, return_onesided=False, scaling='density')
    else:
        raise ValueError(method)
    order = np.argsort(f)
    return f[order], np.maximum(np.real(power[order]), 0)


def metrics(f, p):
    total = p.sum()
    if total <= 0:
        raise ValueError('zero spectral power')
    cdf = np.cumsum(p) / total
    result = {}
    for fraction in (0.90, 0.95, 0.99):
        tail = (1 - fraction) / 2
        low = np.interp(tail, cdf, f)
        high = np.interp(1 - tail, cdf, f)
        result[f'obw{int(fraction*100)}_mhz'] = (high - low) / 1e6
    result['oob_percent'] = 100 * p[np.abs(f) > BW / 2].sum() / total
    return result


def evm_percent(iq, refs, nfft, cp_len):
    hop = nfft + cp_len
    total_error = 0.0
    total_reference = 0.0
    # Exclude first and last OFDM symbols; no signal slicing into partial FFT windows.
    for k in range(1, len(refs) - 1):
        start = k * hop + cp_len
        if start + nfft > len(iq):
            break
        received = np.fft.fftshift(np.fft.fft(iq[start:start + nfft]))
        target = refs[k]
        active = np.abs(target) > 1e-12
        total_error += np.abs(received[active] - target[active]).__pow__(2).sum()
        total_reference += np.abs(target[active]).__pow__(2).sum()
    return 100 * np.sqrt(total_error / total_reference)


records = []
for nfft, overlap in CONFIGS.items():
    print(f'\n===== OFDM_{nfft} (WOLA overlap={overlap}) =====')
    for seed in SEEDS:
        xs, refs, cp_len = make_symbols(nfft, N, BW, seed)
        waves = {
            'original': make_waveform(xs, cp_len, 0)[:N],
            'wola': make_waveform(xs, cp_len, overlap)[:N],
        }
        for mode, iq in waves.items():
            evm = evm_percent(iq, refs, nfft, cp_len)
            for method in ('STFT512', 'Welch8192'):
                f, p = estimate_spectrum(iq, method)
                row = {'nfft': nfft, 'seed': seed, 'mode': mode,
                       'method': method, 'overlap': overlap if mode == 'wola' else 0,
                       **metrics(f, p), 'evm_percent': evm}
                records.append(row)
                print(f"seed={seed} {mode:8s} {method:9s} "
                      f"OBW99={row['obw99_mhz']:.4f} MHz "
                      f"OOB={row['oob_percent']:.3f}% "
                      f"EVM={evm:.6f}%")

fieldnames = list(records[0].keys())
with (OUT_DIR / 'wola_stability_raw.csv').open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

summary = []
for nfft in CONFIGS:
    for method in ('STFT512', 'Welch8192'):
        for mode in ('original', 'wola'):
            group = [r for r in records if r['nfft'] == nfft and
                     r['method'] == method and r['mode'] == mode]
            row = {'nfft': nfft, 'method': method, 'mode': mode}
            for key in ('obw90_mhz', 'obw95_mhz', 'obw99_mhz',
                        'oob_percent', 'evm_percent'):
                values = np.array([r[key] for r in group], dtype=float)
                row[f'{key}_mean'] = float(np.mean(values))
                row[f'{key}_std'] = float(np.std(values, ddof=1))
                row[f'{key}_min'] = float(np.min(values))
                row[f'{key}_max'] = float(np.max(values))
            summary.append(row)

with (OUT_DIR / 'wola_stability_summary.csv').open('w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    writer.writeheader()
    writer.writerows(summary)

for nfft in CONFIGS:
    for method in ('STFT512', 'Welch8192'):
        a = next(r for r in summary if r['nfft'] == nfft and r['method'] == method and r['mode'] == 'original')
        b = next(r for r in summary if r['nfft'] == nfft and r['method'] == method and r['mode'] == 'wola')
        obw_reduction = 100 * (a['obw99_mhz_mean'] - b['obw99_mhz_mean']) / a['obw99_mhz_mean']
        oob_reduction = a['oob_percent_mean'] - b['oob_percent_mean']
        print(f"\nOFDM_{nfft} {method}: "
              f"original OBW99={a['obw99_mhz_mean']:.4f}±{a['obw99_mhz_std']:.4f}, "
              f"WOLA={b['obw99_mhz_mean']:.4f}±{b['obw99_mhz_std']:.4f} MHz")
        print(f"OBW99 reduction={obw_reduction:.2f}%; "
              f"OOB original={a['oob_percent_mean']:.3f}%, "
              f"WOLA={b['oob_percent_mean']:.3f}%, "
              f"decrease={oob_reduction:.3f} percentage points; "
              f"WOLA EVM max={b['evm_percent_max']:.6f}%")

for nfft in CONFIGS:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, ('STFT512', 'Welch8192')):
        for mode in ('original', 'wola'):
            series = [r for r in records if r['nfft'] == nfft and
                      r['method'] == method and r['mode'] == mode]
            x = np.arange(len(SEEDS))
            ax.plot(x, [r['obw99_mhz'] for r in series], marker='o', label=mode)
        ax.set_xticks(np.arange(len(SEEDS)), [str(s)[-2:] for s in SEEDS])
        ax.set_xlabel('Seed (last two digits)')
        ax.set_ylabel('OBW99 (MHz)')
        ax.set_title(method)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f'OFDM_{nfft}: OBW99 stability across seeds')
    fig.tight_layout()
    fig.savefig(OUT_DIR / f'ofdm_{nfft}_stability.png', dpi=160)
    plt.close(fig)

print('\nStability diagnosis completed.')
print('Output directory:', OUT_DIR)
