"""Scale-invariant SDR and permutation-invariant scoring for BSS."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def si_sdr(
    estimate: np.ndarray,
    reference: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Scale-invariant signal-to-distortion ratio in dB."""
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    n = min(est.size, ref.size)
    est = est[:n] - est[:n].mean()
    ref = ref[:n] - ref[:n].mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    num = np.dot(target, target)
    den = np.dot(noise, noise) + eps
    return float(10.0 * np.log10((num + eps) / den))


def si_sdri(
    estimate: np.ndarray,
    reference: np.ndarray,
    mixture: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """SI-SDR improvement: SI-SDR(estimate) − SI-SDR(mixture), in dB."""
    return si_sdr(estimate, reference, eps=eps) - si_sdr(mixture, reference, eps=eps)


def permute_si_sdr(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Match estimated sources to references by maximum mean SI-SDR.

    If ``mixture`` is given, returned scores are SI-SDRi
    (SI-SDR(est) − SI-SDR(mix)) in reference order.
    """
    estimates = np.asarray(estimates)
    references = np.asarray(references)
    n_src = references.shape[0]
    best_mean = -np.inf
    best_perm: tuple[int, ...] = tuple(range(n_src))
    best_scores = np.full(n_src, -np.inf)
    for perm in permutations(range(n_src)):
        scores = np.array(
            [si_sdr(estimates[j], references[i]) for i, j in enumerate(perm)]
        )
        mean = float(np.mean(scores))
        if mean > best_mean:
            best_mean = mean
            best_perm = perm
            best_scores = scores
    if mixture is not None:
        mix_scores = np.array(
            [si_sdr(mixture, references[i]) for i in range(n_src)]
        )
        best_scores = best_scores - mix_scores
        best_mean = float(np.mean(best_scores))
    return best_scores, best_perm, best_mean


def permute_si_sdri(
    estimates: np.ndarray,
    references: np.ndarray,
    mixture: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...], float]:
    """Permutation-invariant SI-SDRi versus the unprocessed mixture."""
    return permute_si_sdr(estimates, references, mixture=mixture)
