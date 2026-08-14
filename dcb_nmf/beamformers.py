"""MVDR and LCMV beamformers with diagonal loading."""

from __future__ import annotations

import numpy as np


def estimate_scm(
    X: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Weighted spatial covariance. X: (F, T, M), mask: (F, T) -> (F, M, M)."""
    w = np.maximum(np.asarray(mask, dtype=np.float64), 0.0)
    w_sum = w.sum(axis=1, keepdims=True) + eps
    w = w / w_sum
    phi = np.einsum("ft,ftm,ftn->fmn", w, X, np.conj(X))
    mics = X.shape[-1]
    phi = phi + eps * np.eye(mics, dtype=np.complex128)[None, :, :]
    phi = 0.5 * (phi + np.conj(np.swapaxes(phi, -1, -2)))
    return phi


def principal_steering(phi: np.ndarray) -> np.ndarray:
    """Principal eigenvector of each SCM, phase-aligned to microphone 0.

    phi : (F, M, M)  ->  a : (F, M)
    """
    _evals, evecs = np.linalg.eigh(phi)
    a = evecs[:, :, -1]
    ref = a[:, :1]
    ref = ref / (np.abs(ref) + 1e-12)
    a = a * np.conj(ref)
    return a


def _regularize(phi: np.ndarray, loading: float) -> np.ndarray:
    mics = phi.shape[-1]
    traces = np.trace(phi, axis1=1, axis2=2).real
    scale = np.maximum(np.mean(traces) / mics, 1e-12)
    return phi + loading * scale * np.eye(mics, dtype=np.complex128)[None, :, :]


def mvdr_weights(
    phi_n: np.ndarray,
    steering: np.ndarray,
    loading: float = 1e-3,
) -> np.ndarray:
    """w(f) = Φ_n^{-1} a / (a^H Φ_n^{-1} a). Returns (F, M)."""
    phi = _regularize(phi_n, loading)
    a = steering
    w_un = np.linalg.solve(phi, a[..., None])[..., 0]
    denom = np.einsum("fm,fm->f", np.conj(a), w_un).real
    denom = np.maximum(denom, 1e-12)
    return w_un / denom[:, None]


def lcmv_weights(
    phi_n: np.ndarray,
    a_target: np.ndarray,
    a_interf: np.ndarray,
    extra: np.ndarray | None = None,
    gain: np.ndarray | None = None,
    loading: float = 1e-3,
) -> np.ndarray:
    """LCMV: min w^H Φ_n w s.t. C^H w = g.

    ``a_interf`` is one interferer steering (F, M). ``extra`` may add more
    interferers with shape (F, M, K) to null in a cocktail-party scene.
    Default g = [1, 0, ..., 0].
    """
    cols = [a_target, a_interf]
    if extra is not None:
        extra = np.asarray(extra)
        if extra.ndim == 2:
            cols.append(extra)
        else:
            cols.extend([extra[:, :, k] for k in range(extra.shape[-1])])
    c = np.stack(cols, axis=-1)
    n_const = c.shape[-1]
    n_mics = a_target.shape[1]
    if n_const > n_mics:
        c = c[:, :, :n_mics]
        n_const = n_mics
    if gain is None:
        gain = np.zeros(n_const, dtype=np.complex128)
        gain[0] = 1.0
    phi = _regularize(phi_n, loading)
    phi_inv_c = np.linalg.solve(phi, c)
    gram = np.einsum("fma,fmb->fab", np.conj(c), phi_inv_c)
    gram = gram + 1e-6 * np.eye(n_const, dtype=np.complex128)[None, :, :]
    n_freq = a_target.shape[0]
    g = np.broadcast_to(np.asarray(gain, dtype=np.complex128)[:n_const], (n_freq, n_const))
    gram_inv_g = np.linalg.solve(gram, g[..., None])[..., 0]
    return np.einsum("fmk,fk->fm", phi_inv_c, gram_inv_g)


def apply_beamformer(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Y(f,t) = w(f)^H x(f,t). X: (F, T, M), weights: (F, M) -> (F, T)."""
    return np.einsum("fm,ftm->ft", np.conj(weights), X)


def ds_weights(steering: np.ndarray) -> np.ndarray:
    """Delay-and-sum: w = a / (a^H a)."""
    a = steering
    denom = np.einsum("fm,fm->f", np.conj(a), a).real
    denom = np.maximum(denom, 1e-12)
    return a / denom[:, None]


def mpdr_weights(
    phi_xx: np.ndarray,
    steering: np.ndarray,
    loading: float = 1e-3,
) -> np.ndarray:
    """Minimum-power distortionless response (Capon) using mixture SCM."""
    return mvdr_weights(phi_xx, steering, loading=loading)


def gev_weights(
    phi_s: np.ndarray,
    phi_n: np.ndarray,
    loading: float = 1e-3,
) -> np.ndarray:
    """Max-SNR GEV beamformer: principal eigenvector of Φ_n^{-1} Φ_s."""
    phi_n = _regularize(phi_n, loading)
    phi_s = 0.5 * (phi_s + np.conj(np.swapaxes(phi_s, -1, -2)))
    a = np.linalg.solve(phi_n, phi_s)
    evals, evecs = np.linalg.eig(a)
    idx = np.argmax(evals.real, axis=-1)
    w = evecs[np.arange(evecs.shape[0]), :, idx]
    # Banse–Warsitz: normalize to unit response at reference mic phase
    ref = w[:, :1]
    ref = ref / (np.abs(ref) + 1e-12)
    w = w * np.conj(ref)
    # Distortionless-ish scaling on mean gain
    gain = np.mean(np.abs(w), axis=1, keepdims=True) + 1e-12
    return w / gain


def mwf_weights(
    phi_s: np.ndarray,
    phi_n: np.ndarray,
    ref_mic: int = 0,
    loading: float = 1e-3,
) -> np.ndarray:
    """Rank-1 multichannel Wiener: w = (Φ_s+Φ_n)^{-1} Φ_s e_ref."""
    phi_n = _regularize(phi_n, loading)
    phi_s = 0.5 * (phi_s + np.conj(np.swapaxes(phi_s, -1, -2)))
    phi_xx = phi_s + phi_n
    e = np.zeros(phi_s.shape[-1], dtype=np.complex128)
    e[ref_mic] = 1.0
    target = phi_s @ e
    return np.linalg.solve(phi_xx, target[..., None])[..., 0]


def zf_weights(
    a_target: np.ndarray,
    a_interfs: list[np.ndarray],
) -> np.ndarray:
    """Zero-forcing / projection: LCMV with identity noise covariance."""
    mics = a_target.shape[1]
    phi_i = np.broadcast_to(
        np.eye(mics, dtype=np.complex128),
        (a_target.shape[0], mics, mics),
    ).copy()
    if not a_interfs:
        return ds_weights(a_target)
    extra = None
    if len(a_interfs) > 1:
        extra = np.stack(a_interfs[1:], axis=-1)
    return lcmv_weights(phi_i, a_target, a_interfs[0], extra=extra, loading=1e-6)
