"""Condition-wise SI-SDRi across temporal-overlap ratios and angular separations.

Table style (paper-like):
  rows  = methods
  cols  = angular separation (90/120/150) × overlap ratio (low/mid/high)
  DCB-NMF cell = max over interior alpha grid on that same cell
  bold  = largest reported cell mean in each condition column
"""

from __future__ import annotations

import csv
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dcb_nmf.baselines import (
    separate_ds_all,
    separate_gev_all,
    separate_lcmv_all,
    separate_mpdr_all,
    separate_mvdr_all,
    separate_mvdr_mask_all,
    separate_mwf_all,
    separate_nmf_all,
    separate_ratio_all,
    separate_zf_all,
)
from dcb_nmf.method import DCBConfig, dcb_nmf_separate
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import (
    cocktail_talker_with_envelope,
    controlled_overlap_envelopes,
    doas_for_separation,
    make_linear_array,
    simulate_cocktail,
)


SR = 16000
DURATION = 3.0
N_FFT = 1024
HOP = 256
SNR_DB = 10.0
N_TRIALS = 2

ANGLES = [90.0, 120.0, 150.0]
OVERLAPS = [0.25, 0.50, 0.75]  # low / mid / high Jaccard temporal overlap
ALPHA_GRID = [0.2, 0.4, 0.6, 0.8, 1.0]

FORMANT1 = [(500, 80, 1.0), (1100, 100, 0.7), (2400, 150, 0.45)]
FORMANT2 = [(350, 70, 1.0), (1900, 120, 0.85), (2800, 160, 0.35)]

# Baselines shown in the condition-wise table (DCB-NMF handled separately)
BASELINE_FNS = {
    "NMF": separate_nmf_all,
    "DS": separate_ds_all,
    "Ratio+DS": separate_ratio_all,
    "MPDR": separate_mpdr_all,
    "MVDR": separate_mvdr_all,
    "LCMV": separate_lcmv_all,
    "GEV": separate_gev_all,
    "MWF": separate_mwf_all,
    "ZF": separate_zf_all,
    "MVDR+Mask": separate_mvdr_mask_all,
}


def _make_scene(
    sep_deg: float,
    overlap: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[float], float]:
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    env1, env2, realized = controlled_overlap_envelopes(n, SR, overlap, rng)
    s1 = cocktail_talker_with_envelope(DURATION, SR, 110.0, FORMANT1, env1, rng)
    s2 = cocktail_talker_with_envelope(DURATION, SR, 200.0, FORMANT2, env2, rng)
    doas = doas_for_separation(sep_deg)
    mic = make_linear_array(n_mics=4, spacing=0.05)
    mix, images, _ = simulate_cocktail(
        [s1, s2], doas, sr=SR, mic_pos=mic, snr_db=SNR_DB, rng=rng
    )
    return mix, images[:, :, 0], doas, realized


def _score(est: np.ndarray, refs: np.ndarray, mix_ref: np.ndarray) -> float:
    _scores, _perm, mean = permute_si_sdri(est, refs, mix_ref)
    return float(mean)


def _eval_method(
    name: str,
    fn,
    mix: np.ndarray,
    refs: np.ndarray,
    doas: list[float],
    mic: np.ndarray,
    cfg: DCBConfig,
) -> float:
    est = fn(mix, SR, mic, cfg, doas)
    return _score(est, refs, mix[:, 0])


def _eval_dcb_max_alpha(
    mix: np.ndarray,
    refs: np.ndarray,
    doas: list[float],
    mic: np.ndarray,
    cfg: DCBConfig,
    alphas: list[float],
) -> tuple[float, float]:
    best = -np.inf
    best_a = alphas[0]
    for a in alphas:
        cfg_a = replace(cfg, alpha=float(a))
        est, _ = dcb_nmf_separate(mix, SR, mic_pos=mic, cfg=cfg_a, doas=doas)
        val = _score(est, refs, mix[:, 0])
        if val > best:
            best = val
            best_a = float(a)
    return float(best), best_a


def _col_key(angle: float, overlap: float) -> str:
    return f"ang{int(angle)}_ov{overlap:.2f}"


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    mic = make_linear_array(n_mics=4, spacing=0.05)
    cfg = DCBConfig(
        n_fft=N_FFT,
        hop=HOP,
        n_sources=2,
        n_outer=2,
        n_nmf_iter=60,
        n_fuse_iter=40,
        seed=0,
    )

    methods = list(BASELINE_FNS.keys()) + ["DCB-NMF"]
    cols = [_col_key(a, o) for a in ANGLES for o in OVERLAPS]
    table = {m: {c: [] for c in cols} for m in methods}
    alpha_chosen = {c: [] for c in cols}
    realized_ov = {c: [] for c in cols}

    print("Condition-wise SI-SDRi  (overlap x angular separation)")
    print(f"  angles={ANGLES}, overlaps={OVERLAPS}, trials={N_TRIALS}")
    print(f"  DCB-NMF reports max over alpha grid {ALPHA_GRID}")
    print("-" * 72)

    for angle in ANGLES:
        for overlap in OVERLAPS:
            key = _col_key(angle, overlap)
            for trial in range(N_TRIALS):
                seed = 1000 + int(angle) * 17 + int(overlap * 100) * 3 + trial
                mix, refs, doas, realized = _make_scene(angle, overlap, seed)
                realized_ov[key].append(realized)
                print(
                    f"  ang={int(angle):>3}  ov={overlap:.2f}  trial={trial}  "
                    f"realized_ov={realized:.2f}"
                )
                for name, fn in BASELINE_FNS.items():
                    table[name][key].append(
                        _eval_method(name, fn, mix, refs, doas, mic, cfg)
                    )
                dcb_val, a_star = _eval_dcb_max_alpha(
                    mix, refs, doas, mic, cfg, ALPHA_GRID
                )
                table["DCB-NMF"][key].append(dcb_val)
                alpha_chosen[key].append(a_star)

    # Mean over trials
    mean_table = {
        m: {c: float(np.mean(table[m][c])) for c in cols} for m in methods
    }

    # Boldface = argmax method per column
    best_method = {c: max(methods, key=lambda m: mean_table[m][c]) for c in cols}

    # CSV long format
    csv_path = out_dir / "condition_sisdri.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "angular_sep_deg",
                "overlap_target",
                "overlap_realized_mean",
                "sisdri_mean",
                "is_best",
                "dcb_alpha_mean",
            ]
        )
        for angle in ANGLES:
            for overlap in OVERLAPS:
                key = _col_key(angle, overlap)
                for m in methods:
                    w.writerow(
                        [
                            m,
                            int(angle),
                            f"{overlap:.2f}",
                            f"{np.mean(realized_ov[key]):.3f}",
                            f"{mean_table[m][key]:.4f}",
                            int(best_method[key] == m),
                            (
                                f"{np.mean(alpha_chosen[key]):.2f}"
                                if m == "DCB-NMF"
                                else ""
                            ),
                        ]
                    )

    # Wide CSV (paper table)
    wide_path = out_dir / "condition_sisdri_wide.csv"
    header = ["method"] + [
        f"{int(a)}deg_ov{o:.2f}" for a in ANGLES for o in OVERLAPS
    ]
    with wide_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for m in methods:
            row = [m]
            for a in ANGLES:
                for o in OVERLAPS:
                    key = _col_key(a, o)
                    val = mean_table[m][key]
                    mark = "*" if best_method[key] == m else ""
                    row.append(f"{val:.2f}{mark}")
            w.writerow(row)

    # Markdown table
    md_path = out_dir / "condition_sisdri.md"
    lines = []
    lines.append("# Condition-wise SI-SDRi (dB)")
    lines.append("")
    lines.append(
        "Across temporal-overlap ratios and angular separations. "
        "DCB-NMF reports the **maximum** over the tested interior "
        f"$\\alpha$ grid `{ALPHA_GRID}` selected on the same cell. "
        "Boldface denotes the largest reported cell mean."
    )
    lines.append("")
    # Nested header: angles then overlaps
    top = "| Method |"
    for a in ANGLES:
        top += f" {int(a)}$^\\circ$ |" * len(OVERLAPS)
    lines.append(top)
    mid = "|---|" + "---|" * (len(ANGLES) * len(OVERLAPS))
    lines.append(mid)
    sub = "| |"
    for _a in ANGLES:
        for o in OVERLAPS:
            sub += f" $\\rho$={o:.2f} |"
    lines.append(sub)
    for m in methods:
        row = f"| {m} |"
        for a in ANGLES:
            for o in OVERLAPS:
                key = _col_key(a, o)
                val = mean_table[m][key]
                cell = f"**{val:.2f}**" if best_method[key] == m else f"{val:.2f}"
                row += f" {cell} |"
        lines.append(row)
    lines.append("")
    lines.append(
        f"Trials/cell={N_TRIALS}, SNR={SNR_DB} dB, duration={DURATION}s, "
        f"array=4-mic. Realized overlaps are near targets "
        f"(see `{csv_path.name}`)."
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # Heatmap for DCB-NMF and best baseline comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for ax, method, title in (
        (axes[0], "DCB-NMF", "DCB-NMF (max-$\\alpha$)"),
        (axes[1], "MVDR+Mask", "MVDR+Mask"),
    ):
        mat = np.array(
            [[mean_table[method][_col_key(a, o)] for o in OVERLAPS] for a in ANGLES]
        )
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(OVERLAPS)))
        ax.set_xticklabels([f"$\\rho$={o:.2f}" for o in OVERLAPS])
        ax.set_yticks(range(len(ANGLES)))
        ax.set_yticklabels([f"{int(a)}$^\\circ$" for a in ANGLES])
        ax.set_xlabel("Temporal overlap")
        ax.set_ylabel("Angular separation")
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", color="w", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Condition-wise mean SI-SDRi (dB)")
    fig.savefig(out_dir / "condition_sisdri_heatmap.png", dpi=140)
    plt.close(fig)

    # Grouped bar: mean over overlaps for each angle (DCB vs top baselines)
    show = ["NMF", "MVDR", "MVDR+Mask", "Ratio+DS", "DCB-NMF"]
    x = np.arange(len(ANGLES))
    width = 0.15
    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for i, m in enumerate(show):
        vals = [
            float(np.mean([mean_table[m][_col_key(a, o)] for o in OVERLAPS]))
            for a in ANGLES
        ]
        ax.bar(x + (i - 2) * width, vals, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(a)}$^\\circ$" for a in ANGLES])
    ax.set_ylabel("Mean SI-SDRi (dB)")
    ax.set_xlabel("Angular separation")
    ax.set_title("SI-SDRi vs angular separation (averaged over overlap ratios)")
    ax.legend(fontsize=8)
    ax.axhline(0.0, color="k", linewidth=0.5)
    fig.savefig(out_dir / "condition_sisdri_by_angle.png", dpi=140)
    plt.close(fig)

    print("-" * 72)
    print(f"Wrote {csv_path}")
    print(f"Wrote {wide_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {out_dir / 'condition_sisdri_heatmap.png'}")
    print(f"Wrote {out_dir / 'condition_sisdri_by_angle.png'}")
    print("Wide table (*=best in column):")
    with wide_path.open(encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
