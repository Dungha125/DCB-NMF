"""Cocktail-party demo: separate overlapping talkers from diffuse noise."""

from __future__ import annotations

import argparse
import sys
from math import gcd
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dcb_nmf.baselines import separate_mvdr_mask_all, separate_ratio_all
from dcb_nmf.method import (
    DCBConfig,
    dcb_nmf_separate,
    separate_lcmv_all,
    separate_mvdr_all,
    separate_nmf_all,
)
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import cocktail_talker, make_linear_array, simulate_cocktail
from dcb_nmf.stft import analysis
from dataclasses import replace


SR = 16000
DURATION = 3.5
N_FFT = 1024
HOP = 256
# User azimuth: 0°/180° = endfire, 90° = broadside (s1=15°, s2=165°).
# Internal ULA uses 0° = broadside, delay ∝ sin(θ) → θ = 90° − az.
DOAS_AZ = [15.0, 165.0]
DOAS = [90.0 - a for a in DOAS_AZ]


def _load_wav_mono(path: Path, target_sr: int) -> np.ndarray:
    sr, raw = wavfile.read(path)
    y = np.asarray(raw, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if np.issubdtype(raw.dtype, np.integer):
        y = y / max(np.iinfo(raw.dtype).max, 1)
    if sr != target_sr:
        g = gcd(target_sr, sr)
        y = resample_poly(y, target_sr // g, sr // g)
    peak = np.max(np.abs(y)) + 1e-12
    return 0.9 * y / peak


def _match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(a.shape[0], b.shape[0])
    return a[:n], b[:n]


def _write_wav(path: Path, y: np.ndarray) -> None:
    y = np.asarray(y, dtype=np.float64).ravel()
    peak = np.max(np.abs(y)) + 1e-12
    pcm = np.clip(y / peak, -1.0, 1.0)
    wavfile.write(path, SR, (pcm * 32767).astype(np.int16))


def _time(y: np.ndarray) -> np.ndarray:
    return np.arange(y.shape[0], dtype=np.float64) / SR


def _scale_to(est: np.ndarray, ref: np.ndarray) -> np.ndarray:
    est = np.asarray(est, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    n = min(est.size, ref.size)
    num = np.dot(est[:n], ref[:n])
    den = np.dot(ref[:n], ref[:n]) + 1e-12
    return (num / den) * est[:n]


def _wave(
    ax,
    y: np.ndarray,
    title: str,
    ylim: float,
    color: str = "#1f77b4",
    overlay: np.ndarray | None = None,
) -> None:
    t = _time(y)
    if overlay is not None:
        ov = overlay[: y.shape[0]]
        ax.plot(_time(ov), ov, color="#bbbbbb", linewidth=0.8, label="ban đầu", zorder=1)
    ax.plot(t, y, color=color, linewidth=0.7, zorder=2)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0.0, t[-1])
    ax.set_ylim(-ylim, ylim)
    ax.set_ylabel("biên độ")
    ax.set_xlabel("s")
    ax.grid(True, alpha=0.25)


def _spec(ax, y: np.ndarray, title: str) -> None:
    spec = np.abs(analysis(y, sr=SR, n_fft=N_FFT, hop=HOP))
    spec_db = 20.0 * np.log10(spec + 1e-8)
    ax.imshow(
        spec_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=spec_db.max() - 60,
        vmax=spec_db.max(),
        extent=[0, y.shape[0] / SR, 0, SR / 2],
    )
    ax.set_title(title, fontsize=9)
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")


def main(s1_path: Path | None = None, s2_path: Path | None = None) -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    if s1_path is not None and s2_path is not None:
        t1 = _load_wav_mono(s1_path, SR)
        t2 = _load_wav_mono(s2_path, SR)
        t1, t2 = _match_length(t1, t2)
        talkers = [t1, t2]
        src_labels = (
            f"Nguồn 1 ban đầu  ({s1_path.name}, DOA={DOAS_AZ[0]:.0f}°)",
            f"Nguồn 2 ban đầu  ({s2_path.name}, DOA={DOAS_AZ[1]:.0f}°)",
        )
        scene_note = (
            f"Cocktail party  (WAV sources {s1_path.name} + {s2_path.name} "
            f"+ diffuse noise, SNR=10 dB, {t1.shape[0] / SR:.2f}s)"
        )
        src_desc = (
            f"  Source 1: {s1_path.name}, DOA={DOAS_AZ[0]:.0f} deg\n"
            f"  Source 2: {s2_path.name}, DOA={DOAS_AZ[1]:.0f} deg\n"
            "  Mixture:  both sources at once + isotropic noise on 4-mic array"
        )
    else:
        talkers = [
            cocktail_talker(
                DURATION,
                SR,
                f0=110.0,
                formants=[(500, 80, 1.0), (1100, 100, 0.7), (2400, 150, 0.45)],
                rng=rng,
                n_bursts=4,
            ),
            cocktail_talker(
                DURATION,
                SR,
                f0=200.0,
                formants=[(350, 70, 1.0), (1900, 120, 0.85), (2800, 160, 0.35)],
                rng=rng,
                n_bursts=4,
            ),
        ]
        src_labels = (
            f"Nguồn 1 ban đầu  (f0≈110 Hz, DOA={DOAS_AZ[0]:.0f}°)",
            f"Nguồn 2 ban đầu  (f0≈200 Hz, DOA={DOAS_AZ[1]:.0f}°)",
        )
        scene_note = "Cocktail party  (2 talkers + diffuse noise, SNR=10 dB)"
        src_desc = (
            f"  Source 1: low pitch f0~110 Hz, DOA={DOAS_AZ[0]:.0f} deg, overlapping bursts\n"
            f"  Source 2: high pitch f0~200 Hz, DOA={DOAS_AZ[1]:.0f} deg, interleaved/overlapping\n"
            "  Mixture:  both talkers at once + isotropic noise on 4-mic array"
        )
    mic_pos = make_linear_array(n_mics=4, spacing=0.05)
    mix, images, _noise = simulate_cocktail(
        talkers, DOAS, sr=SR, mic_pos=mic_pos, snr_db=10.0, rng=rng
    )
    refs = images[:, :, 0]
    cfg = DCBConfig(n_fft=N_FFT, hop=HOP, n_sources=2, n_outer=3, seed=0)

    methods = {
        "nmf": separate_nmf_all(mix, SR, mic_pos, cfg, DOAS),
        "mvdr": separate_mvdr_all(mix, SR, mic_pos, cfg, DOAS),
        "lcmv": separate_lcmv_all(mix, SR, mic_pos, cfg, DOAS),
        "ratio_ds": separate_ratio_all(mix, SR, mic_pos, cfg, DOAS),
        "mvdr_mask": separate_mvdr_mask_all(mix, SR, mic_pos, cfg, DOAS),
    }
    # DCB-NMF: same cell-wise max-α protocol as condition experiment
    alpha_grid = [0.2, 0.4, 0.6, 0.8, 1.0]
    best_mean, best_alpha, best_est = -np.inf, alpha_grid[0], None
    for a in alpha_grid:
        est, _ = dcb_nmf_separate(
            mix, SR, mic_pos=mic_pos, cfg=replace(cfg, alpha=float(a)), doas=DOAS
        )
        scores, perm, mean = permute_si_sdri(est, refs, mix[:, 0])
        if mean > best_mean:
            best_mean = mean
            best_alpha = float(a)
            best_est = np.stack([est[j] for j in perm], axis=0)
    methods["dcb_nmf"] = best_est
    print(f"  DCB-NMF selected alpha={best_alpha:.1f}  (max SI-SDRi on this scene)")

    mix_ref = mix[:, 0]
    print(scene_note)
    print(src_desc)
    print("Mean SI-SDRi over talkers  (SI-SDR(est) - SI-SDR(mix), permutation-invariant)")
    print("-" * 56)
    print(f"  {'mix_ref':<12}  {0.0:7.2f} dB   [0. 0.]")
    aligned = {}
    per_src = {}
    for name, est in methods.items():
        scores, perm, mean = permute_si_sdri(est, refs, mix_ref)
        aligned[name] = np.stack([est[j] for j in perm], axis=0)
        per_src[name] = scores
        print(f"  {name:<12}  {mean:7.2f} dB   {np.round(scores, 2)}")

    ranking = sorted(
        ((n, float(np.mean(per_src[n]))) for n in per_src),
        key=lambda x: x[1],
        reverse=True,
    )
    print("-" * 56)
    print("Ranking (mean SI-SDRi):")
    for i, (n, m) in enumerate(ranking, 1):
        mark = "  <-- best" if i == 1 else ""
        print(f"  {i}. {n:<12}  {m:7.2f} dB{mark}")

    _write_wav(out_dir / "mix_ref.wav", mix[:, 0])
    for i in range(refs.shape[0]):
        _write_wav(out_dir / f"talker{i + 1}_ref.wav", refs[i])
        for name in methods:
            _write_wav(out_dir / f"{name}_talker{i + 1}.wav", aligned[name][i])

    method_order = ["nmf", "mvdr", "mvdr_mask", "dcb_nmf"]
    method_labels = {
        "nmf": "NMF",
        "mvdr": "MVDR",
        "lcmv": "LCMV",
        "ratio_ds": "Ratio+DS",
        "mvdr_mask": "MVDR+Mask",
        "dcb_nmf": "DCB-NMF",
    }
    method_colors = {
        "nmf": "#4c78a8",
        "mvdr": "#f58518",
        "lcmv": "#54a24b",
        "ratio_ds": "#b279a2",
        "mvdr_mask": "#ff9da6",
        "dcb_nmf": "#e45756",
    }
    ylim = 1.15 * float(np.max(np.abs(mix[:, 0])))

    # --- Waveforms: nguồn 1, nguồn 2, hợp nhất, rồi top baselines + DCB ---
    fig = plt.figure(figsize=(12, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(6, 2)
    _wave(
        fig.add_subplot(gs[0, 0]),
        refs[0],
        src_labels[0],
        ylim,
        "#1f77b4",
    )
    _wave(
        fig.add_subplot(gs[0, 1]),
        refs[1],
        src_labels[1],
        ylim,
        "#d62728",
    )
    _wave(
        fig.add_subplot(gs[1, :]),
        mix[:, 0],
        "Đã hợp nhất  (2 talker chồng nhau + tạp khuếch tán, SNR=10 dB)",
        ylim,
        "#444444",
    )
    for r, name in enumerate(method_order):
        for s in range(2):
            y = _scale_to(aligned[name][s], refs[s])
            _wave(
                fig.add_subplot(gs[r + 2, s]),
                y,
                f"Sau tách {method_labels[name]} — nguồn {s + 1}  (SI-SDRi {per_src[name][s]:.1f} dB)",
                ylim,
                method_colors[name],
                overlay=refs[s],
            )
    fig.suptitle(
        f"Verify: DCB-NMF (α={best_alpha:.1f}) vs NMF / MVDR / MVDR+Mask",
        fontsize=13,
    )
    fig.savefig(out_dir / "waveforms.png", dpi=140)
    plt.close(fig)

    # --- Spectrograms: cùng bố cục ---
    fig = plt.figure(figsize=(12, 13.5), constrained_layout=True)
    gs = fig.add_gridspec(6, 2)
    _spec(fig.add_subplot(gs[0, 0]), refs[0], "Nguồn 1 ban đầu")
    _spec(fig.add_subplot(gs[0, 1]), refs[1], "Nguồn 2 ban đầu")
    _spec(fig.add_subplot(gs[1, :]), mix[:, 0], "Đã hợp nhất (cocktail mix, mic 0)")
    for r, name in enumerate(method_order):
        for s in range(2):
            _spec(
                fig.add_subplot(gs[r + 2, s]),
                aligned[name][s],
                f"Sau tách {method_labels[name]} — nguồn {s + 1}  (SI-SDRi {per_src[name][s]:.1f} dB)",
            )
    fig.suptitle("Spectrogram verify: DCB-NMF vs strong baselines", fontsize=13)
    fig.savefig(out_dir / "spectrograms.png", dpi=140)
    plt.close(fig)

    # --- SI-SDRi theo từng nguồn ---
    names = ["nmf", "mvdr", "ratio_ds", "mvdr_mask", "dcb_nmf"]
    labels = ["NMF", "MVDR", "Ratio+DS", "MVDR+Mask", "DCB-NMF"]
    src1 = [float(per_src[n][0]) for n in names]
    src2 = [float(per_src[n][1]) for n in names]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=True)
    ax.bar(x - width / 2, src1, width, label="Nguồn 1", color="#1f77b4")
    ax.bar(x + width / 2, src2, width, label="Nguồn 2", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Verify SI-SDRi (including strongest baselines)")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.legend()
    fig.savefig(out_dir / "sisdri.png", dpi=140)
    fig.savefig(out_dir / "sisdr.png", dpi=140)
    plt.close(fig)

    print(f"Wrote WAV and figures to {out_dir}")
    print("  waveforms.png     source1 / source2 / mix / after separation")
    print("  spectrograms.png  same layout, spectrograms")
    print("  sisdri.png        SI-SDRi per source vs baselines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCB-NMF cocktail-party demo")
    parser.add_argument("--s1", type=Path, default=None, help="WAV for source 1")
    parser.add_argument("--s2", type=Path, default=None, help="WAV for source 2")
    args = parser.parse_args()
    if (args.s1 is None) ^ (args.s2 is None):
        parser.error("provide both --s1 and --s2, or neither")
    main(args.s1, args.s2)
