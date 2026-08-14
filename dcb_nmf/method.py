"""Closed-loop Dual Complementary Beamforming NMF (DCB-NMF)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .beamformers import (
    apply_beamformer,
    estimate_scm,
    lcmv_weights,
    mvdr_weights,
    principal_steering,
)
from .mix import phase_align, scan_doa, steering_vectors
from .nmf import cluster_bases, fuse_magnitudes, kl_nmf, wiener_masks
from .stft import analysis, synthesis


@dataclass
class DCBConfig:
    n_fft: int = 1024
    hop: int = 256
    n_bases: int = 16
    n_fuse_bases: int = 10
    n_nmf_iter: int = 80
    n_fuse_iter: int = 60
    n_outer: int = 3
    n_sources: int = 2
    ref_mic: int = 0
    loading: float = 2e-2
    lcmv_align_max: float = 0.92
    # Blend after NMF fusion: mag = α·fused + (1-α)·(mask·|Y_mvdr|)
    alpha: float = 0.5
    seed: int = 0


def _nmf_init(
    mag: np.ndarray,
    cfg: DCBConfig,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    w, h = kl_nmf(mag, n_bases=cfg.n_bases, n_iter=cfg.n_nmf_iter, rng=rng)
    labels = cluster_bases(w, h, n_clusters=cfg.n_sources, rng=rng)
    masks = wiener_masks(w, h, labels, n_sources=cfg.n_sources)
    recon = []
    for s in range(cfg.n_sources):
        idx = np.where(labels == s)[0]
        if idx.size == 0:
            recon.append(np.zeros_like(mag))
        else:
            recon.append(w[:, idx] @ h[idx, :])
    return masks, recon


def _identify_target(
    X: np.ndarray,
    masks: list[np.ndarray],
    sr: int,
    n_fft: int,
    target_doa: float | None,
    mic_pos: np.ndarray | None,
) -> int:
    """Pick the source whose TF bins match the target inter-mic phase."""
    if target_doa is None or mic_pos is None or X.shape[-1] < 2:
        energies = [float(np.mean(m)) for m in masks]
        return int(np.argmax(energies))

    a_th = steering_vectors(n_fft, sr, mic_pos, target_doa)[: X.shape[0]]
    ipd_th = np.angle(a_th[:, 1] * np.conj(a_th[:, 0]))
    ipd = np.angle(X[:, :, 1] * np.conj(X[:, :, 0]))
    best_idx, best_score = 0, -np.inf
    for i, mask in enumerate(masks):
        weight = mask / (mask.sum() + 1e-12)
        score = float(np.sum(weight * np.cos(ipd - ipd_th[:, None])))
        if score > best_score:
            best_idx, best_score = i, score
    return best_idx


def _steering_alignment(a_s: np.ndarray, a_i: np.ndarray) -> float:
    num = np.abs(np.einsum("fm,fm->f", np.conj(a_s), a_i))
    den = np.linalg.norm(a_s, axis=1) * np.linalg.norm(a_i, axis=1) + 1e-12
    return float(np.mean(num / den))


def _manifold_steering(
    phi: np.ndarray,
    sr: int,
    n_fft: int,
    mic_pos: np.ndarray | None,
    doa_hint: float | None = None,
) -> np.ndarray:
    """Array-manifold steering: scan DOA from SCM, or use a provided hint."""
    if mic_pos is None:
        return principal_steering(phi)
    theta = doa_hint if doa_hint is not None else scan_doa(phi, mic_pos, sr, n_fft)
    steer = steering_vectors(n_fft, sr, mic_pos, theta)[: phi.shape[0]]
    return phase_align(steer)


def _beamform_source(
    X: np.ndarray,
    mask_s: np.ndarray,
    other_masks: list[np.ndarray],
    loading: float,
    lcmv_align_max: float,
    sr: int,
    n_fft: int,
    mic_pos: np.ndarray | None,
    target_doa: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """MVDR + LCMV for one cocktail-party source; others are interferers."""
    mask_n = np.zeros_like(mask_s)
    for m in other_masks:
        mask_n = mask_n + m
    mask_n = np.clip(mask_n, 0.0, 1.0)
    if float(mask_n.mean()) < 1e-6:
        mask_n = np.clip(1.0 - mask_s, 0.0, 1.0)
    phi_s = estimate_scm(X, mask_s)
    phi_n = estimate_scm(X, mask_n)
    a_s = _manifold_steering(phi_s, sr, n_fft, mic_pos, doa_hint=target_doa)
    w_mvdr = mvdr_weights(phi_n, a_s, loading=loading)
    y_mvdr = apply_beamformer(X, w_mvdr)

    interferers = []
    for m in other_masks:
        if float(m.mean()) < 1e-5:
            continue
        a_i = _manifold_steering(estimate_scm(X, m), sr, n_fft, mic_pos)
        if _steering_alignment(a_s, a_i) < lcmv_align_max:
            interferers.append(a_i)
    n_mics = X.shape[-1]
    max_nulls = max(1, n_mics - 1)
    interferers = interferers[: max_nulls - 1] if interferers else []
    if interferers:
        extra = None
        if len(interferers) > 1:
            extra = np.stack(interferers[1:], axis=-1)
        w_lcmv = lcmv_weights(
            phi_n, a_s, interferers[0], extra=extra, loading=loading
        )
        y_lcmv = apply_beamformer(X, w_lcmv)
        return y_mvdr, y_lcmv, True
    return y_mvdr, y_mvdr, False


def _beamform_pair(
    X: np.ndarray,
    mask_s: np.ndarray,
    mask_i: np.ndarray,
    loading: float,
    lcmv_align_max: float,
    sr: int,
    n_fft: int,
    mic_pos: np.ndarray | None,
    target_doa: float | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    return _beamform_source(
        X,
        mask_s,
        [mask_i],
        loading,
        lcmv_align_max,
        sr,
        n_fft,
        mic_pos,
        target_doa,
    )


def dcb_nmf_separate(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Separate every overlapping source (cocktail-party BSS).

    Returns estimates of shape (n_sources, n_samples).
    """
    cfg = cfg or DCBConfig()
    rng = np.random.default_rng(cfg.seed)
    n_samples = mix.shape[0]
    X = analysis(mix, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop)
    mag_ref = np.abs(X[:, :, cfg.ref_mic])

    masks, recon = _nmf_init(mag_ref, cfg, rng)
    n_src = cfg.n_sources
    if doas is not None and mic_pos is not None:
        order = []
        remaining = list(range(n_src))
        for doa in doas[:n_src]:
            pick = _identify_target(
                X, [masks[i] for i in remaining], sr, cfg.n_fft, doa, mic_pos
            )
            order.append(remaining.pop(pick))
        order.extend(remaining)
        masks = [masks[i] for i in order]
        recon = [recon[i] for i in order]

    priors = [np.maximum(r, 1e-10) for r in recon]
    fused_specs = [None] * n_src
    last_mvdr = [None] * n_src
    last_lcmv = [None] * n_src
    used_lcmv = False

    for _ in range(cfg.n_outer):
        mags = []
        for k in range(n_src):
            others = [masks[j] for j in range(n_src) if j != k]
            hint = doas[k] if doas is not None and k < len(doas) else None
            y_mvdr, y_lcmv, ok = _beamform_source(
                X,
                masks[k],
                others,
                cfg.loading,
                cfg.lcmv_align_max,
                sr,
                cfg.n_fft,
                mic_pos,
                hint,
            )
            used_lcmv = used_lcmv or ok
            masked_mvdr = masks[k] * np.abs(y_mvdr)
            views = [np.abs(y_mvdr), priors[k], masked_mvdr]
            consensus = (0, 1, 2)
            if ok:
                views.append(np.abs(y_lcmv))
            mag = fuse_magnitudes(
                views,
                n_bases=cfg.n_fuse_bases,
                n_iter=cfg.n_fuse_iter,
                rng=rng,
                consensus_idx=consensus,
            )
            alpha = float(np.clip(cfg.alpha, 0.0, 1.0))
            mag = alpha * mag + (1.0 - alpha) * masked_mvdr
            fused_specs[k] = mag * np.exp(1j * np.angle(y_mvdr))
            last_mvdr[k] = y_mvdr
            last_lcmv[k] = y_lcmv
            mags.append(mag)
        total = sum(mags) + 1e-10
        masks = [m / total for m in mags]
        priors = mags

    estimates = np.stack(
        [
            synthesis(spec, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=n_samples)
            for spec in fused_specs
        ],
        axis=0,
    )
    extras = {
        "stft": X,
        "masks": masks,
        "used_lcmv": used_lcmv,
        "y_mvdr": last_mvdr,
        "y_lcmv": last_lcmv,
        "fused": fused_specs,
    }
    return estimates, extras


def dcb_nmf(
    mix: np.ndarray,
    sr: int,
    target_doa: float | None = None,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Extract one target from a cocktail mix (or source 0 if no DOA)."""
    cfg = cfg or DCBConfig()
    doas = [target_doa] if target_doa is not None else None
    ys, extras = dcb_nmf_separate(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)
    return ys[0], extras


def _init_masks(
    mix: np.ndarray,
    sr: int,
    cfg: DCBConfig,
    mic_pos: np.ndarray | None,
    doas: list[float] | None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(cfg.seed)
    X = analysis(mix, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop)
    mag_ref = np.abs(X[:, :, cfg.ref_mic])
    masks, _recon = _nmf_init(mag_ref, cfg, rng)
    if doas is not None and mic_pos is not None:
        order = []
        remaining = list(range(len(masks)))
        for doa in doas[: len(masks)]:
            pick = _identify_target(
                X, [masks[i] for i in remaining], sr, cfg.n_fft, doa, mic_pos
            )
            order.append(remaining.pop(pick))
        order.extend(remaining)
        masks = [masks[i] for i in order]
    return X, masks


def separate_nmf_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Wiener NMF for every source. Returns (n_sources, n_samples)."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    ref = X[:, :, cfg.ref_mic]
    return np.stack(
        [
            synthesis(m * ref, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=mix.shape[0])
            for m in masks
        ],
        axis=0,
    )


def separate_mvdr_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """One-shot NMF-mask MVDR for every source."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k, mask in enumerate(masks):
        others = [masks[j] for j in range(len(masks)) if j != k]
        hint = doas[k] if doas is not None and k < len(doas) else None
        y_mvdr, _y, _ok = _beamform_source(
            X,
            mask,
            others,
            cfg.loading,
            cfg.lcmv_align_max,
            sr,
            cfg.n_fft,
            mic_pos,
            hint,
        )
        outs.append(
            synthesis(y_mvdr, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=mix.shape[0])
        )
    return np.stack(outs, axis=0)


def separate_lcmv_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """One-shot NMF-mask LCMV for every source."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k, mask in enumerate(masks):
        others = [masks[j] for j in range(len(masks)) if j != k]
        hint = doas[k] if doas is not None and k < len(doas) else None
        _y, y_lcmv, _ok = _beamform_source(
            X,
            mask,
            others,
            cfg.loading,
            cfg.lcmv_align_max,
            sr,
            cfg.n_fft,
            mic_pos,
            hint,
        )
        outs.append(
            synthesis(y_lcmv, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=mix.shape[0])
        )
    return np.stack(outs, axis=0)


def separate_nmf(
    mix: np.ndarray,
    sr: int,
    target_doa: float | None = None,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
) -> np.ndarray:
    """Single-channel Wiener NMF on the reference microphone."""
    doas = [target_doa] if target_doa is not None else None
    return separate_nmf_all(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)[0]


def separate_mvdr(
    mix: np.ndarray,
    sr: int,
    target_doa: float | None = None,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
) -> np.ndarray:
    """One-shot NMF-mask MVDR (no LCMV, no fusion loop)."""
    doas = [target_doa] if target_doa is not None else None
    return separate_mvdr_all(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)[0]


def separate_lcmv(
    mix: np.ndarray,
    sr: int,
    target_doa: float | None = None,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
) -> np.ndarray:
    """One-shot NMF-mask LCMV (falls back to MVDR if steering vectors collapse)."""
    doas = [target_doa] if target_doa is not None else None
    return separate_lcmv_all(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)[0]
