"""Far-field linear-array mixing for synthetic multi-channel demos."""

from __future__ import annotations

import numpy as np

SOUND_SPEED = 343.0


def make_linear_array(n_mics: int = 4, spacing: float = 0.05) -> np.ndarray:
    """Microphone x-coordinates (metres), centered at the origin."""
    pos = (np.arange(n_mics) - (n_mics - 1) / 2.0) * spacing
    return pos.astype(np.float64)


def mic_x_from_xyz(xyz) -> np.ndarray:
    """Centered x-coordinates of a linear array given XYZ positions."""
    pos = np.asarray(xyz, dtype=np.float64)
    x = pos[:, 0]
    return x - x.mean()


def azimuth_to_broadside_doa(az_deg: float) -> float:
    """Map dataset azimuth (0° = +x endfire, 90° = +y broadside) to ULA DOA.

    ``steering_vectors`` uses τ = p sin(θ)/c with θ = 0 at broadside.
    Dataset angles are measured from the array axis, so θ = 90° − az.
    """
    return 90.0 - float(az_deg)


def phase_align(steering: np.ndarray) -> np.ndarray:
    """Make microphone 0 real and nonnegative at each frequency."""
    ref = steering[:, :1]
    ref = ref / (np.abs(ref) + 1e-12)
    return steering * np.conj(ref)


def scan_doa(
    phi: np.ndarray,
    mic_pos: np.ndarray,
    sr: int,
    n_fft: int,
    grid: np.ndarray | None = None,
    c: float = SOUND_SPEED,
) -> float:
    """DOA (degrees) maximizing average steering power a^H Φ a."""
    if grid is None:
        grid = np.linspace(-90.0, 90.0, 181)
    n_freq = phi.shape[0]
    best_theta = 0.0
    best_power = -np.inf
    for theta in grid:
        steer = phase_align(
            steering_vectors(n_fft, sr, mic_pos, float(theta), c=c)[:n_freq]
        )
        power = np.einsum("fm,fmn,fn->", np.conj(steer), phi, steer).real
        if power > best_power:
            best_power = power
            best_theta = float(theta)
    return best_theta


def steering_vectors(
    n_fft: int,
    sr: int,
    mic_pos: np.ndarray,
    doa_deg: float,
    c: float = SOUND_SPEED,
) -> np.ndarray:
    """Far-field steering a(f, m) = exp(-j 2π f τ_m), τ_m = p_m sin(θ) / c.

    Returns (n_freq, n_mics).
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    delay = mic_pos * np.sin(np.deg2rad(doa_deg)) / c
    return np.exp(-1j * 2.0 * np.pi * freqs[:, None] * delay[None, :])


def harmonic_utterance(
    duration: float,
    sr: int,
    f0: float,
    formants: list[tuple[float, float, float]],
    syllable_hz: float,
    rng: np.random.Generator,
    n_harmonics: int = 24,
) -> np.ndarray:
    """Speech-like harmonic source with formant envelope and syllable AM."""
    n = int(duration * sr)
    t = np.arange(n, dtype=np.float64) / sr
    sig = np.zeros(n, dtype=np.float64)
    nyquist = sr / 2.0
    for h in range(1, n_harmonics + 1):
        freq = h * f0
        if freq >= nyquist:
            break
        gain = 0.0
        for center, bandwidth, amp in formants:
            gain += amp * np.exp(-0.5 * ((freq - center) / bandwidth) ** 2)
        jitter = 0.003 * rng.normal() * f0
        sig += gain * np.sin(2.0 * np.pi * (freq + jitter) * t)
    env = 0.55 + 0.45 * np.sin(2.0 * np.pi * syllable_hz * t + rng.uniform(0, 2 * np.pi))
    env = np.clip(env, 0.0, 1.0) ** 1.8
    noise = rng.normal(0.0, 0.02, size=n)
    sig = sig * env + noise * env
    peak = np.max(np.abs(sig)) + 1e-12
    return 0.9 * sig / peak


def burst_envelope(
    n: int,
    sr: int,
    rng: np.random.Generator,
    n_bursts: int = 4,
    min_s: float = 0.35,
    max_s: float = 0.95,
) -> np.ndarray:
    """On/off phrase envelope so talkers overlap like a cocktail party."""
    env = np.zeros(n, dtype=np.float64)
    for _ in range(n_bursts):
        length = int(rng.uniform(min_s, max_s) * sr)
        start = int(rng.integers(0, max(1, n - length)))
        end = min(n, start + length)
        env[start:end] += np.hanning(end - start)
    peak = np.max(env) + 1e-12
    return np.clip(env / peak, 0.0, 1.0)


def controlled_overlap_envelopes(
    n: int,
    sr: int,
    overlap_ratio: float,
    rng: np.random.Generator,
    duty: float = 0.55,
    fade_s: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build two activity envelopes with target Jaccard temporal overlap.

    overlap_ratio ≈ |A∩B| / |A∪B| for binary activity.
    Returns (env1, env2, realized_overlap).
    """
    overlap_ratio = float(np.clip(overlap_ratio, 0.05, 0.95))
    duty = float(np.clip(duty, 0.35, 0.8))
    active = int(duty * n)
    # Jaccard ρ = I / (2A - I) with equal activity lengths A
    # => I = ρ * 2A / (1+ρ)
    inter = int(round(overlap_ratio * 2 * active / (1.0 + overlap_ratio)))
    inter = int(np.clip(inter, 1, active))
    union = 2 * active - inter
    # Layout on timeline: [only1 | both | only2] packed near center with jitter
    only1 = active - inter
    only2 = active - inter
    pack = only1 + inter + only2
    if pack > n:
        scale = n / pack
        only1 = max(1, int(only1 * scale))
        only2 = max(1, int(only2 * scale))
        inter = max(1, int(inter * scale))
        pack = only1 + inter + only2
    start = max(0, (n - pack) // 2 + int(rng.integers(-n // 20, n // 20 + 1)))
    start = int(np.clip(start, 0, max(0, n - pack)))
    a = np.zeros(n, dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    i0 = start
    i1 = start + only1
    i2 = start + only1 + inter
    i3 = start + pack
    a[i0:i2] = 1.0
    b[i1:i3] = 1.0
    fade = max(1, int(fade_s * sr))
    for env in (a, b):
        on = np.flatnonzero(env > 0.5)
        if on.size == 0:
            continue
        left, right = int(on[0]), int(on[-1]) + 1
        fl = min(fade, max(1, (right - left) // 4))
        env[left : left + fl] *= np.linspace(0.0, 1.0, fl)
        env[right - fl : right] *= np.linspace(1.0, 0.0, fl)
    both = (a > 0.1) & (b > 0.1)
    either = (a > 0.1) | (b > 0.1)
    realized = float(both.sum() / max(either.sum(), 1))
    return a, b, realized


def cocktail_talker_with_envelope(
    duration: float,
    sr: int,
    f0: float,
    formants: list[tuple[float, float, float]],
    envelope: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Speech-like source multiplied by a provided activity envelope."""
    voiced = harmonic_utterance(
        duration,
        sr,
        f0,
        formants,
        syllable_hz=float(rng.uniform(2.2, 4.0)),
        rng=rng,
    )
    env = np.asarray(envelope, dtype=np.float64)
    if env.shape[0] != voiced.shape[0]:
        env = np.interp(
            np.linspace(0, 1, voiced.shape[0]),
            np.linspace(0, 1, env.shape[0]),
            env,
        )
    return voiced * env


def cocktail_talker(
    duration: float,
    sr: int,
    f0: float,
    formants: list[tuple[float, float, float]],
    rng: np.random.Generator,
    n_bursts: int = 4,
) -> np.ndarray:
    """Speech-like source with overlapping bursts (cocktail-party talker)."""
    voiced = harmonic_utterance(
        duration,
        sr,
        f0,
        formants,
        syllable_hz=float(rng.uniform(2.2, 4.0)),
        rng=rng,
    )
    n = voiced.shape[0]
    return voiced * burst_envelope(n, sr, rng, n_bursts=n_bursts)


def doas_for_separation(sep_deg: float) -> list[float]:
    """Symmetric DOA pair with given angular separation (degrees)."""
    half = 0.5 * float(sep_deg)
    return [-half, half]


def isotropic_noise(
    n_samples: int,
    sr: int,
    mic_pos: np.ndarray,
    rng: np.random.Generator,
    n_waves: int = 16,
    c: float = SOUND_SPEED,
) -> np.ndarray:
    """Approximate spherically isotropic noise as many far-field noise waves."""
    n_mics = mic_pos.shape[0]
    noise = np.zeros((n_samples, n_mics), dtype=np.float64)
    for theta in np.linspace(-90.0, 90.0, n_waves):
        wave = rng.normal(size=n_samples)
        # mild pink tilt
        spec = np.fft.rfft(wave)
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
        spec *= 1.0 / np.maximum(np.sqrt(freqs), 20.0)
        wave = np.fft.irfft(spec, n=n_samples).real
        delays = mic_pos * np.sin(np.deg2rad(theta)) / c
        for m, tau in enumerate(delays):
            noise[:, m] += _fractional_delay(wave, float(tau), sr)
    rms = np.sqrt(np.mean(noise[:, 0] ** 2)) + 1e-12
    return noise / rms


def simulate_cocktail(
    sources: list[np.ndarray],
    doas_deg: list[float],
    sr: int,
    mic_pos: np.ndarray,
    snr_db: float = 10.0,
    c: float = SOUND_SPEED,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cocktail-party mix: equal-power overlapping talkers + diffuse noise.

    Returns mix (n_samples, n_mics), dry images (n_src, n_samples, n_mics),
    and the diffuse-noise field (n_samples, n_mics).
    """
    rng = rng or np.random.default_rng()
    n_samples = sources[0].shape[0]
    n_mics = mic_pos.shape[0]
    n_src = len(sources)
    images = np.zeros((n_src, n_samples, n_mics), dtype=np.float64)

    for s, (src, doa) in enumerate(zip(sources, doas_deg)):
        delays = mic_pos * np.sin(np.deg2rad(doa)) / c
        for m, tau in enumerate(delays):
            images[s, :, m] = _fractional_delay(src, float(tau), sr)

    ref_pow = np.mean(images[0, :, 0] ** 2) + 1e-12
    for s in range(1, n_src):
        p = np.mean(images[s, :, 0] ** 2) + 1e-12
        images[s] *= np.sqrt(ref_pow / p)

    speech = images.sum(axis=0)
    noise = isotropic_noise(n_samples, sr, mic_pos, rng, c=c)
    speech_pow = np.mean(speech[:, 0] ** 2) + 1e-12
    noise *= np.sqrt(speech_pow) * 10 ** (-snr_db / 20.0)
    mix = speech + noise
    return mix, images, noise


def _fractional_delay(x: np.ndarray, delay_s: float, sr: int) -> np.ndarray:
    """All-pass fractional delay via an FFT phase ramp."""
    n = x.shape[0]
    n_fft = 1 << int(np.ceil(np.log2(max(n, 8))))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    shifted = np.fft.irfft(
        np.fft.rfft(x, n=n_fft) * np.exp(-2j * np.pi * freqs * delay_s),
        n=n_fft,
    )
    return shifted[:n].real


def simulate_array(
    sources: list[np.ndarray],
    doas_deg: list[float],
    sr: int,
    mic_pos: np.ndarray,
    n_fft: int = 1024,
    hop: int = 256,
    sir_db: float = 0.0,
    snr_db: float = 20.0,
    c: float = SOUND_SPEED,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix sources onto a far-field array.

    Returns
    -------
    mix : (n_samples, n_mics)
    images : (n_sources, n_samples, n_mics) dry spatial images before sensor noise
    """
    del n_fft, hop  # mixing is time-domain; kept for call-site compatibility
    rng = rng or np.random.default_rng()
    n_samples = sources[0].shape[0]
    n_mics = mic_pos.shape[0]
    n_src = len(sources)
    images = np.zeros((n_src, n_samples, n_mics), dtype=np.float64)

    for s, (src, doa) in enumerate(zip(sources, doas_deg)):
        delays = mic_pos * np.sin(np.deg2rad(doa)) / c
        for m, tau in enumerate(delays):
            images[s, :, m] = _fractional_delay(src, float(tau), sr)

    target_pow = np.mean(images[0, :, 0] ** 2) + 1e-12
    if n_src > 1:
        interf_pow = np.mean(images[1, :, 0] ** 2) + 1e-12
        scale = np.sqrt(target_pow / interf_pow) * 10 ** (-sir_db / 20.0)
        images[1] *= scale

    mix = images.sum(axis=0)
    sig_pow = np.mean(mix[:, 0] ** 2) + 1e-12
    noise_std = np.sqrt(sig_pow) * 10 ** (-snr_db / 20.0)
    mix = mix + rng.normal(0.0, noise_std, size=mix.shape)
    return mix, images
