"""Evaluate DCB-NMF on the HN (LRS2-2mix spatial) dataset.

Protocol
--------
1. Sweep α on a small held-in subset (first N files / angle config).
2. Pick the global α with the best mean SI-SDRi, and the per-config α.
3. Run NMF / MWF / MVDR+Mask / Ratio+DS / DCB-NMF on every mix with the chosen α.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dcb_nmf.baselines import (
    separate_mwf_all,
    separate_mvdr_mask_all,
    separate_ratio_all,
)
from dcb_nmf.method import DCBConfig, dcb_nmf_separate, separate_nmf_all
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import azimuth_to_broadside_doa, mic_x_from_xyz

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


SR = 16000
ALPHA_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
METHODS = ("NMF", "MWF", "MVDR+Mask", "Ratio+DS", "DCB-NMF")


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    sr, raw = wavfile.read(path)
    y = np.asarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        y = y / max(np.iinfo(raw.dtype).max, 1)
    return int(sr), y


def _mono(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2:
        return y[:, 0]
    return y.ravel()


def _discover_configs(data_root: Path) -> list[dict]:
    rows = []
    for folder in sorted(data_root.glob("config_*")):
        if not folder.is_dir():
            continue
        meta = json.loads((folder / "config_metadata.json").read_text(encoding="utf-8"))
        ang = meta["angle_config"]
        xyz = meta.get("mic_positions_xyz") or []
        rows.append(
            {
                "folder": folder,
                "name": folder.name,
                "config": int(ang["config"]),
                "sep": float(ang["sep"]),
                "s1_az": float(ang["s1_angle"]),
                "s2_az": float(ang["s2_angle"]),
                "mic_pos": mic_x_from_xyz(xyz),
                "doas": [
                    azimuth_to_broadside_doa(ang["s1_angle"]),
                    azimuth_to_broadside_doa(ang["s2_angle"]),
                ],
            }
        )
    if not rows:
        raise FileNotFoundError(f"no config_* folders under {data_root}")
    return rows


def _list_mixes(cfg: dict) -> list[Path]:
    return sorted((cfg["folder"] / "mix").glob("*.wav"))


def _load_example(cfg: dict, mix_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    name = mix_path.name
    sr_m, mix = _load_wav(mix_path)
    sr_1, s1 = _load_wav(cfg["folder"] / "s1" / name)
    sr_2, s2 = _load_wav(cfg["folder"] / "s2" / name)
    if sr_m != SR or sr_1 != SR or sr_2 != SR:
        raise ValueError(f"expected {SR} Hz, got {sr_m}/{sr_1}/{sr_2} in {name}")
    if mix.ndim != 2:
        raise ValueError(f"mix must be multi-channel, got {mix.shape}")
    refs = np.stack([_mono(s1), _mono(s2)], axis=0)
    n = min(mix.shape[0], refs.shape[1])
    return mix[:n], refs[:, :n], mix[:n, 0]


def _cfg() -> DCBConfig:
    return DCBConfig(
        n_fft=1024,
        hop=256,
        n_sources=2,
        n_outer=3,
        n_nmf_iter=80,
        n_fuse_iter=60,
        seed=0,
    )


def _run_method(
    name: str,
    mix: np.ndarray,
    mic_pos: np.ndarray,
    doas: list[float],
    cfg: DCBConfig,
) -> np.ndarray:
    if name == "NMF":
        return separate_nmf_all(mix, SR, mic_pos, cfg, doas)
    if name == "MWF":
        return separate_mwf_all(mix, SR, mic_pos, cfg, doas)
    if name == "MVDR+Mask":
        return separate_mvdr_mask_all(mix, SR, mic_pos, cfg, doas)
    if name == "Ratio+DS":
        return separate_ratio_all(mix, SR, mic_pos, cfg, doas)
    if name == "DCB-NMF":
        est, _ = dcb_nmf_separate(mix, SR, mic_pos=mic_pos, cfg=cfg, doas=doas)
        return est
    raise KeyError(name)


def _score(est: np.ndarray, refs: np.ndarray, mix_ref: np.ndarray) -> tuple[np.ndarray, float]:
    scores, _perm, mean = permute_si_sdri(est, refs, mix_ref)
    return scores, float(mean)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _tune_alpha(
    configs: list[dict],
    n_tune: int,
    out_dir: Path,
) -> tuple[float, dict[str, float], list[dict]]:
    cfg0 = _cfg()
    rows: list[dict] = []
    print("Alpha sweep")
    print(f"  grid={ALPHA_GRID}  files/config={n_tune}")
    print("-" * 72)
    for spec in configs:
        files = _list_mixes(spec)[:n_tune]
        for mix_path in files:
            mix, refs, mix_ref = _load_example(spec, mix_path)
            for a in ALPHA_GRID:
                est, _ = dcb_nmf_separate(
                    mix,
                    SR,
                    mic_pos=spec["mic_pos"],
                    cfg=replace(cfg0, alpha=float(a)),
                    doas=spec["doas"],
                )
                scores, mean = _score(est, refs, mix_ref)
                row = {
                    "config": spec["name"],
                    "sep_deg": f"{spec['sep']:.0f}",
                    "filename": mix_path.name,
                    "alpha": f"{a:.1f}",
                    "sisdri_mean": f"{mean:.4f}",
                    "sisdri_s1": f"{scores[0]:.4f}",
                    "sisdri_s2": f"{scores[1]:.4f}",
                }
                rows.append(row)
                print(
                    f"  {spec['name']}  {mix_path.name[:28]:<28}  "
                    f"a={a:.1f}  {mean:6.2f} dB"
                )
    _write_csv(
        out_dir / "alpha_tune.csv",
        rows,
        ["config", "sep_deg", "filename", "alpha", "sisdri_mean", "sisdri_s1", "sisdri_s2"],
    )

    by_alpha: dict[float, list[float]] = {a: [] for a in ALPHA_GRID}
    by_cfg_alpha: dict[str, dict[float, list[float]]] = {
        spec["name"]: {a: [] for a in ALPHA_GRID} for spec in configs
    }
    for row in rows:
        a = float(row["alpha"])
        val = float(row["sisdri_mean"])
        by_alpha[a].append(val)
        by_cfg_alpha[row["config"]][a].append(val)

    mean_by_alpha = {a: float(np.mean(v)) for a, v in by_alpha.items()}
    best_global = max(mean_by_alpha, key=mean_by_alpha.get)
    best_per_cfg = {
        name: max(vals, key=lambda a: float(np.mean(vals[a])))
        for name, vals in by_cfg_alpha.items()
    }

    print("-" * 72)
    print("Mean SI-SDRi vs alpha (tune set):")
    for a in ALPHA_GRID:
        mark = "  <-- best" if a == best_global else ""
        print(f"  a={a:.1f}  {mean_by_alpha[a]:6.2f} dB{mark}")
    print("Per-config best alpha:")
    for spec in configs:
        a = best_per_cfg[spec["name"]]
        print(
            f"  {spec['name']}: a={a:.1f}  "
            f"({np.mean(by_cfg_alpha[spec['name']][a]):.2f} dB)"
        )

    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.plot(ALPHA_GRID, [mean_by_alpha[a] for a in ALPHA_GRID], "o-", color="#e45756")
    ax.axvline(best_global, color="k", linewidth=0.8, linestyle="--")
    ax.set_xlabel("α")
    ax.set_ylabel("Mean SI-SDRi (dB)")
    ax.set_title("DCB-NMF α sweep on HN tune set")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_dir / "alpha_tune.png", dpi=140)
    plt.close(fig)
    return best_global, best_per_cfg, rows


def _eval_all(
    configs: list[dict],
    alpha_global: float,
    alpha_per_cfg: dict[str, float],
    out_dir: Path,
    limit: int | None,
    use_per_cfg_alpha: bool,
) -> list[dict]:
    cfg0 = _cfg()
    csv_path = out_dir / "sisdri.csv"
    fieldnames = [
        "config",
        "sep_deg",
        "s1_az",
        "s2_az",
        "filename",
        "method",
        "alpha",
        "sisdri_mean",
        "sisdri_s1",
        "sisdri_s2",
    ]
    done: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                done.add((row["config"], row["filename"], row["method"]))

    jobs = []
    for spec in configs:
        files = _list_mixes(spec)
        if limit is not None:
            files = files[:limit]
        for mix_path in files:
            jobs.append((spec, mix_path))

    total = len(jobs)
    print("Full evaluation")
    print(f"  files={total}  methods={list(METHODS)}")
    print("-" * 72)

    def flush() -> None:
        _write_csv(csv_path, rows, fieldnames)

    for i, (spec, mix_path) in enumerate(jobs, 1):
        pending = [
            name
            for name in METHODS
            if (spec["name"], mix_path.name, name) not in done
        ]
        if not pending:
            continue
        mix, refs, mix_ref = _load_example(spec, mix_path)
        alpha = (
            float(alpha_per_cfg[spec["name"]])
            if use_per_cfg_alpha
            else float(alpha_global)
        )
        cfg_dcb = replace(cfg0, alpha=alpha)
        line_bits = []
        for name in METHODS:
            key = (spec["name"], mix_path.name, name)
            if key in done:
                continue
            est = _run_method(name, mix, spec["mic_pos"], spec["doas"], cfg_dcb)
            scores, mean = _score(est, refs, mix_ref)
            rows.append(
                {
                    "config": spec["name"],
                    "sep_deg": f"{spec['sep']:.0f}",
                    "s1_az": f"{spec['s1_az']:.0f}",
                    "s2_az": f"{spec['s2_az']:.0f}",
                    "filename": mix_path.name,
                    "method": name,
                    "alpha": f"{alpha:.1f}" if name == "DCB-NMF" else "",
                    "sisdri_mean": f"{mean:.4f}",
                    "sisdri_s1": f"{scores[0]:.4f}",
                    "sisdri_s2": f"{scores[1]:.4f}",
                }
            )
            done.add(key)
            line_bits.append(f"{name} {mean:5.2f}")
        if line_bits:
            print(f"  [{i:3d}/{total}] {spec['name']}  " + "  ".join(line_bits))
        if i % 5 == 0:
            flush()
    flush()
    return rows


def _summarize(rows: list[dict], out_dir: Path, alpha_global: float) -> None:
    methods = list(METHODS)
    configs = sorted({r["config"] for r in rows})
    seps = {}
    for r in rows:
        seps[r["config"]] = int(float(r["sep_deg"]))

    def mean_of(method: str, config: str | None = None) -> float:
        vals = [
            float(r["sisdri_mean"])
            for r in rows
            if r["method"] == method and (config is None or r["config"] == config)
        ]
        return float(np.mean(vals)) if vals else float("nan")

    summary_rows = []
    for m in methods:
        row = {"method": m, "overall": f"{mean_of(m):.4f}"}
        for c in configs:
            row[c] = f"{mean_of(m, c):.4f}"
        summary_rows.append(row)
    fields = ["method", "overall"] + configs
    _write_csv(out_dir / "summary.csv", summary_rows, fields)

    best = max(methods, key=lambda m: mean_of(m))
    lines = [
        "# HN dataset SI-SDRi (dB)",
        "",
        f"DCB-NMF uses a single global α={alpha_global:.1f} chosen on the tune subset.",
        "References are reverberant spatial images at microphone 0.",
        "",
        "| Method | Overall | " + " | ".join(f"{seps[c]}$^\\circ$" for c in configs) + " |",
        "|---|" + "---|" * (1 + len(configs)),
    ]
    for m in methods:
        cells = []
        for label, val in (("overall", mean_of(m)),) + tuple(
            (c, mean_of(m, c)) for c in configs
        ):
            cell = f"{val:.2f}"
            if m == best and label == "overall":
                cell = f"**{cell}**"
            cells.append(cell)
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    x = np.arange(len(configs))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    colors = {
        "NMF": "#4c78a8",
        "MWF": "#f58518",
        "MVDR+Mask": "#54a24b",
        "Ratio+DS": "#b279a2",
        "DCB-NMF": "#e45756",
    }
    for i, m in enumerate(methods):
        vals = [mean_of(m, c) for c in configs]
        ax.bar(x + (i - 2) * width, vals, width, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{seps[c]}°" for c in configs])
    ax.set_xlabel("Angular separation")
    ax.set_ylabel("Mean SI-SDRi (dB)")
    ax.set_title(f"HN dataset  (DCB-NMF α={alpha_global:.1f})")
    ax.axhline(0.0, color="k", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "sisdri_by_angle.png", dpi=140)
    plt.close(fig)

    print("-" * 72)
    print(f"Wrote {out_dir / 'summary.md'}")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DCB-NMF on HN spatial mixes")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "hn-20260818T164446Z-1-001" / "hn",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "hn")
    parser.add_argument("--tune-files", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="max files per config")
    parser.add_argument("--alpha", type=float, default=None, help="skip sweep, use this α")
    parser.add_argument(
        "--per-config-alpha",
        action="store_true",
        help="use the best α of each angle config instead of one global α",
    )
    args = parser.parse_args()

    data_root = args.data.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = _discover_configs(data_root)
    print(f"HN data: {data_root}")
    print(f"  configs={len(configs)}, files/config={len(_list_mixes(configs[0]))}")
    for spec in configs:
        print(
            f"  {spec['name']}: s1={spec['s1_az']:.0f} deg  s2={spec['s2_az']:.0f} deg  "
            f"sep={spec['sep']:.0f} deg  doas={np.round(spec['doas'], 1)}"
        )

    if args.alpha is not None:
        alpha_global = float(args.alpha)
        alpha_per_cfg = {spec["name"]: alpha_global for spec in configs}
        print(f"Using provided alpha={alpha_global:.1f}")
    else:
        alpha_global, alpha_per_cfg, _tune_rows = _tune_alpha(
            configs, args.tune_files, out_dir
        )

    rows = _eval_all(
        configs,
        alpha_global,
        alpha_per_cfg,
        out_dir,
        args.limit,
        use_per_cfg_alpha=args.per_config_alpha,
    )
    _summarize(rows, out_dir, alpha_global)


if __name__ == "__main__":
    main()
