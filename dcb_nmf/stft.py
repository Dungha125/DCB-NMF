"""STFT / iSTFT helpers for single- and multi-channel audio."""

from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft


def analysis(
    x: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
) -> np.ndarray:
    """Return complex STFT.

    Parameters
    ----------
    x : (n_samples,) or (n_samples, n_mics)
    Returns
    -------
    X : (n_freq, n_frames) or (n_freq, n_frames, n_mics)
    """
    x = np.asarray(x, dtype=np.float64)
    kwargs = dict(
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
    )
    if x.ndim == 1:
        _, _, Z = stft(x, **kwargs)
        return Z
    if x.ndim != 2:
        raise ValueError("x must be 1-D or 2-D (samples, mics)")
    channels = [
        stft(x[:, m], **kwargs)[2] for m in range(x.shape[1])
    ]
    return np.stack(channels, axis=-1)


def synthesis(
    X: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop: int = 256,
    window: str = "hann",
    length: int | None = None,
) -> np.ndarray:
    """Invert an STFT produced by :func:`analysis`."""
    X = np.asarray(X)
    kwargs = dict(
        fs=sr,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        nfft=n_fft,
    )
    if X.ndim == 2:
        _, y = istft(X, **kwargs)
        if length is not None:
            y = _match_length(y, length)
        return y.astype(np.float64)
    if X.ndim != 3:
        raise ValueError("X must be 2-D or 3-D")
    ys = []
    for m in range(X.shape[-1]):
        _, y = istft(X[:, :, m], **kwargs)
        ys.append(y)
    y = np.stack(ys, axis=-1)
    if length is not None:
        y = _match_length(y, length)
    return y.astype(np.float64)


def _match_length(y: np.ndarray, length: int) -> np.ndarray:
    if y.shape[0] >= length:
        return y[:length]
    pad = [(0, length - y.shape[0])] + [(0, 0)] * (y.ndim - 1)
    return np.pad(y, pad)
