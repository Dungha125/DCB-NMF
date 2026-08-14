"""KL-NMF, basis clustering, Wiener masks, and dual-spectrogram fusion."""

from __future__ import annotations

import numpy as np


def kl_nmf(
    V: np.ndarray,
    n_bases: int,
    n_iter: int = 100,
    eps: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """KL-divergence NMF: V ≈ W H with multiplicative updates.

    V : (F, T) nonnegative
    Returns W : (F, K), H : (K, T)
    """
    rng = rng or np.random.default_rng()
    V = np.maximum(np.asarray(V, dtype=np.float64), eps)
    F, T = V.shape
    W = rng.random((F, n_bases)) + eps
    H = rng.random((n_bases, T)) + eps
    W /= W.sum(axis=0, keepdims=True) + eps
    ones = np.ones_like(V)
    for _ in range(n_iter):
        WH = W @ H + eps
        W *= ((V / WH) @ H.T) / (ones @ H.T + eps)
        W = np.maximum(W, eps)
        WH = W @ H + eps
        H *= (W.T @ (V / WH)) / (W.T @ ones + eps)
        H = np.maximum(H, eps)
    return W, H


def cluster_bases(
    W: np.ndarray,
    H: np.ndarray,
    n_clusters: int = 2,
    n_iter: int = 40,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Cluster NMF spectral templates (columns of W) into source groups.

    Concurrent sources share similar activations, so clustering H fails;
    harmonic / formant structure lives in W.
    """
    rng = rng or np.random.default_rng()
    K = W.shape[1]
    energy = H.sum(axis=1)
    if n_clusters >= K:
        return np.arange(K)
    X = W.T
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    labels = _kmeans(X, n_clusters, n_iter=n_iter, rng=rng)
    if _empty_cluster(labels, n_clusters):
        labels = (energy > np.median(energy)).astype(np.int64)
        if labels.min() == labels.max():
            labels = np.zeros(K, dtype=np.int64)
            labels[K // 2 :] = 1
    return labels


def wiener_masks(
    W: np.ndarray,
    H: np.ndarray,
    labels: np.ndarray,
    n_sources: int = 2,
    eps: float = 1e-10,
) -> list[np.ndarray]:
    """Soft Wiener masks, one per clustered source. Each mask is (F, T)."""
    recon = []
    for s in range(n_sources):
        idx = np.where(labels == s)[0]
        if idx.size == 0:
            recon.append(np.zeros((W.shape[0], H.shape[1])))
        else:
            recon.append(W[:, idx] @ H[idx, :])
    total = sum(recon) + eps
    return [r / total for r in recon]


def fuse_magnitudes(
    mags: list[np.ndarray],
    n_bases: int = 8,
    n_iter: int = 80,
    eps: float = 1e-10,
    rng: np.random.Generator | None = None,
    consensus_idx: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Shared-basis NMF fusion of complementary magnitude spectrograms.

    All views are factorized jointly. The returned magnitude is the
    geometric mean of the reconstructions listed in ``consensus_idx``
    (default: every view), so a distorted view can still inform shared
    activations without being averaged into the output.
    """
    views = [np.maximum(np.asarray(m, dtype=np.float64), 0.0) for m in mags]
    n_freq, _n_frames = views[0].shape
    stacked = np.vstack(views)
    w, h = kl_nmf(stacked, n_bases=n_bases, n_iter=n_iter, eps=eps, rng=rng)
    recons = [
        np.maximum(w[i * n_freq : (i + 1) * n_freq] @ h, eps)
        for i in range(len(views))
    ]
    if consensus_idx is None:
        consensus_idx = tuple(range(len(recons)))
    chosen = np.stack([recons[i] for i in consensus_idx], axis=0)
    return np.exp(np.mean(np.log(chosen), axis=0))


def fuse_dual_magnitude(
    mag_a: np.ndarray,
    mag_b: np.ndarray,
    n_bases: int = 8,
    n_iter: int = 80,
    eps: float = 1e-10,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Shared-basis NMF fusion of two complementary magnitude spectrograms."""
    return fuse_magnitudes(
        [mag_a, mag_b], n_bases=n_bases, n_iter=n_iter, eps=eps, rng=rng
    )


def _kmeans(
    X: np.ndarray,
    k: int,
    n_iter: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = X.shape[0]
    centroids = X[rng.choice(n, k, replace=False)].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(n_iter):
        dist = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1)
        for j in range(k):
            members = X[labels == j]
            if members.shape[0] == 0:
                centroids[j] = X[rng.integers(0, n)]
            else:
                c = members.mean(axis=0)
                centroids[j] = c / (np.linalg.norm(c) + 1e-8)
    return labels


def _empty_cluster(labels: np.ndarray, n_clusters: int) -> bool:
    return any(np.sum(labels == s) == 0 for s in range(n_clusters))
