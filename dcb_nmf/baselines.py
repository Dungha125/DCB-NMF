"""Cocktail-party baselines sharing the same NMF masks / DOA pipeline."""

from __future__ import annotations

import numpy as np

from .beamformers import (
    apply_beamformer,
    ds_weights,
    estimate_scm,
    gev_weights,
    mpdr_weights,
    mwf_weights,
    mvdr_weights,
    zf_weights,
)
from .method import (
    DCBConfig,
    _init_masks,
    _manifold_steering,
    _steering_alignment,
    dcb_nmf_separate,
    separate_lcmv_all,
    separate_mvdr_all,
    separate_nmf_all,
)
from .stft import analysis, synthesis


def _source_stats(
    X: np.ndarray,
    masks: list[np.ndarray],
    k: int,
    sr: int,
    cfg: DCBConfig,
    mic_pos: np.ndarray | None,
    doas: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return Φ_s, Φ_n, a_s, list of interferer steerings for source k."""
    others = [masks[j] for j in range(len(masks)) if j != k]
    mask_n = np.zeros_like(masks[k])
    for m in others:
        mask_n = mask_n + m
    mask_n = np.clip(mask_n, 0.0, 1.0)
    if float(mask_n.mean()) < 1e-6:
        mask_n = np.clip(1.0 - masks[k], 0.0, 1.0)
    phi_s = estimate_scm(X, masks[k])
    phi_n = estimate_scm(X, mask_n)
    hint = doas[k] if doas is not None and k < len(doas) else None
    a_s = _manifold_steering(phi_s, sr, cfg.n_fft, mic_pos, doa_hint=hint)
    interfs = []
    for m in others:
        if float(m.mean()) < 1e-5:
            continue
        a_i = _manifold_steering(estimate_scm(X, m), sr, cfg.n_fft, mic_pos)
        if _steering_alignment(a_s, a_i) < cfg.lcmv_align_max:
            interfs.append(a_i)
    return phi_s, phi_n, a_s, interfs


def _synth_all(specs: list[np.ndarray], sr: int, cfg: DCBConfig, n: int) -> np.ndarray:
    return np.stack(
        [synthesis(y, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=n) for y in specs],
        axis=0,
    )


def separate_ds_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Delay-and-sum beamforming steered by NMF-mask DOA."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, ds_weights(a_s)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_mpdr_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """MPDR / Capon using mixture SCM + NMF steering."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    ones = np.ones(X.shape[:2], dtype=np.float64)
    phi_xx = estimate_scm(X, ones)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, mpdr_weights(phi_xx, a_s, cfg.loading)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_gev_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """GEV (max-SNR) beamformer from NMF speech/noise SCMs."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, _a, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, gev_weights(phi_s, phi_n, cfg.loading)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_mwf_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Multichannel Wiener filter from NMF SCMs."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, _a, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(
            apply_beamformer(
                X, mwf_weights(phi_s, phi_n, ref_mic=cfg.ref_mic, loading=cfg.loading)
            )
        )
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_zf_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Zero-forcing spatial nulling of interferers."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, interfs = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        n_mics = X.shape[-1]
        interfs = interfs[: max(0, n_mics - 1)]
        outs.append(apply_beamformer(X, zf_weights(a_s, interfs)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def _auxiva_freq(
    Xf: np.ndarray,
    n_iter: int = 30,
    eps: float = 1e-8,
) -> np.ndarray:
    """AuxIVA iterative projection on one frequency. Xf: (T, M) complex."""
    t_frames, n_mics = Xf.shape
    w = np.eye(n_mics, dtype=np.complex128)
    for _ in range(n_iter):
        y = Xf @ np.conj(w.T)
        r = np.sqrt(np.maximum(np.abs(y) ** 2, eps))
        for n in range(n_mics):
            g = 1.0 / r[:, n]
            v = (Xf.T * g) @ np.conj(Xf) / t_frames
            v = 0.5 * (v + np.conj(v.T)) + eps * np.eye(n_mics)
            w_n = np.linalg.solve(w @ v, np.eye(n_mics)[:, n])
            denom = np.sqrt(np.maximum(np.real(np.conj(w_n) @ v @ w_n), eps))
            w[n] = w_n / denom
    return Xf @ np.conj(w.T)


def separate_mvdr_mask_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """MVDR followed by NMF soft-mask postfilter."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        y = apply_beamformer(X, mvdr_weights(phi_n, a_s, cfg.loading))
        outs.append(masks[k] * y)
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_auxiva_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """PCA + AuxIVA; permute outputs to NMF masks (or DOA order)."""
    cfg = cfg or DCBConfig()
    n_src = cfg.n_sources
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    n_freq, n_frames, _n_mics = X.shape
    ones = np.ones((n_freq, n_frames), dtype=np.float64)
    phi = estimate_scm(X, ones)
    Xp = np.zeros((n_freq, n_frames, n_src), dtype=np.complex128)
    for f in range(n_freq):
        _evals, evecs = np.linalg.eigh(phi[f])
        u = evecs[:, -n_src:]
        Xp[f] = X[f] @ np.conj(u)

    Y = np.zeros_like(Xp)
    for f in range(n_freq):
        Y[f] = _auxiva_freq(Xp[f], n_iter=30)

    # Match AuxIVA channels to NMF sources by TF magnitude correlation
    scores = np.zeros((n_src, n_src))
    for s in range(n_src):
        for c in range(n_src):
            scores[s, c] = float(np.corrcoef(masks[s].ravel(), np.abs(Y[:, :, c]).ravel())[0, 1])
    assign = [-1] * n_src
    score_work = np.nan_to_num(scores, nan=-1.0)
    for _ in range(n_src):
        s, c = np.unravel_index(np.argmax(score_work), score_work.shape)
        assign[int(s)] = int(c)
        score_work[int(s), :] = -np.inf
        score_work[:, int(c)] = -np.inf

    # Wiener-like rescale using mix reference phase energy
    ref = X[:, :, cfg.ref_mic]
    specs = []
    for s in range(n_src):
        y = Y[:, :, assign[s]]
        # take magnitude from IVA, phase from mix; soft-mask with NMF
        mag = np.abs(y)
        specs.append(masks[s] * mag * np.exp(1j * np.angle(ref)))
    return _synth_all(specs, sr, cfg, mix.shape[0])


def separate_ratio_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Soft spectral-ratio mask on delay-and-sum output (NMF prior × DS)."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        y = apply_beamformer(X, ds_weights(a_s))
        outs.append(masks[k] * y)
    return _synth_all(outs, sr, cfg, mix.shape[0])


def run_all_baselines(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Run ~10 cocktail-party separators. Returns name -> (n_src, n_samples)."""
    cfg = cfg or DCBConfig()
    results: dict[str, np.ndarray] = {}
    results["NMF"] = separate_nmf_all(mix, sr, mic_pos, cfg, doas)
    results["DS"] = separate_ds_all(mix, sr, mic_pos, cfg, doas)
    results["Ratio+DS"] = separate_ratio_all(mix, sr, mic_pos, cfg, doas)
    results["MPDR"] = separate_mpdr_all(mix, sr, mic_pos, cfg, doas)
    results["MVDR"] = separate_mvdr_all(mix, sr, mic_pos, cfg, doas)
    results["LCMV"] = separate_lcmv_all(mix, sr, mic_pos, cfg, doas)
    results["GEV"] = separate_gev_all(mix, sr, mic_pos, cfg, doas)
    results["MWF"] = separate_mwf_all(mix, sr, mic_pos, cfg, doas)
    results["ZF"] = separate_zf_all(mix, sr, mic_pos, cfg, doas)
    results["MVDR+Mask"] = separate_mvdr_mask_all(mix, sr, mic_pos, cfg, doas)
    results["AuxIVA"] = separate_auxiva_all(mix, sr, mic_pos, cfg, doas)
    dcb, _ = dcb_nmf_separate(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)
    results["DCB-NMF"] = dcb
    return results
