"""Evaluate ~10 cocktail-party separators with SI-SDRi."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dcb_nmf.baselines import run_all_baselines
from dcb_nmf.method import DCBConfig
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import cocktail_talker, make_linear_array, simulate_cocktail


SR = 16000
DURATION = 3.5
N_FFT = 1024
HOP = 256
DOAS = [-40.0, 45.0]
SNR_DB = 10.0


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

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
    mic_pos = make_linear_array(n_mics=4, spacing=0.05)
    mix, images, _ = simulate_cocktail(
        talkers, DOAS, sr=SR, mic_pos=mic_pos, snr_db=SNR_DB, rng=rng
    )
    refs = images[:, :, 0]
    mix_ref = mix[:, 0]
    cfg = DCBConfig(n_fft=N_FFT, hop=HOP, n_sources=2, n_outer=3, seed=0)

    print("Cocktail-party baseline suite  (2 talkers + diffuse noise)")
    print(f"  SNR={SNR_DB} dB, DOAs={DOAS}, 4-mic array")
    print("SI-SDRi = SI-SDR(est) - SI-SDR(mix)   [permutation-invariant]")
    print("-" * 64)

    results = run_all_baselines(mix, SR, mic_pos, cfg, DOAS)
    rows = []
    for name, est in results.items():
        scores, _perm, mean = permute_si_sdri(est, refs, mix_ref)
        rows.append(
            {
                "method": name,
                "sisdri_src1": float(scores[0]),
                "sisdri_src2": float(scores[1]),
                "sisdri_mean": float(mean),
            }
        )
        print(
            f"  {name:<10}  mean {mean:7.2f} dB   "
            f"[{scores[0]:6.2f}, {scores[1]:6.2f}]"
        )

    rows.sort(key=lambda r: r["sisdri_mean"], reverse=True)
    csv_path = out_dir / "baseline_sisdri.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["method", "sisdri_src1", "sisdri_src2", "sisdri_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    methods = [r["method"] for r in rows]
    means = [r["sisdri_mean"] for r in rows]
    s1 = [r["sisdri_src1"] for r in rows]
    s2 = [r["sisdri_src2"] for r in rows]

    colors = []
    for m in methods:
        colors.append("#e45756" if m == "DCB-NMF" else "#4c78a8")

    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.bar(methods, means, color=colors)
    ax.set_ylabel("Mean SI-SDRi (dB)")
    ax.set_title("Cocktail party: ~10 models ranked by mean SI-SDRi")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.tick_params(axis="x", rotation=30)
    for i, v in enumerate(means):
        ax.text(i, v + 0.15, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(out_dir / "baseline_sisdri.png", dpi=140)
    plt.close(fig)

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, s1, width, label="Source 1", color="#1f77b4")
    ax.bar(x + width / 2, s2, width, label="Source 2", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30)
    ax.set_ylabel("SI-SDRi (dB)")
    ax.set_title("Per-source SI-SDRi across models")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.legend()
    fig.savefig(out_dir / "baseline_sisdri_per_source.png", dpi=140)
    plt.close(fig)

    print("-" * 64)
    print(f"Best: {rows[0]['method']}  ({rows[0]['sisdri_mean']:.2f} dB mean SI-SDRi)")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'baseline_sisdri.png'}")
    print(f"Wrote {out_dir / 'baseline_sisdri_per_source.png'}")


if __name__ == "__main__":
    main()
