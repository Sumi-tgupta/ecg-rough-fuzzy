"""ECG preprocessing: baseline removal, band-pass, notch, noise handling, normalisation."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy import signal as sps
from config import BANDPASS, BASELINE_MEDIAN_WINDOWS, NOTCH_HZ, FS


def resample_to(x: np.ndarray, fs_in: float, fs_out: float = FS) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return x
    frac = Fraction(fs_out / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, frac.numerator, frac.denominator, axis=0)


def remove_baseline(x, fs, windows=BASELINE_MEDIAN_WINDOWS):
    """Two-stage median filter (200 ms, 600 ms) estimates baseline wander; subtract it."""
    out = np.asarray(x, dtype=float)
    base = out
    for w in windows:
        base = sps.medfilt(base, kernel_size=int(w * fs) | 1)
    return out - base


def bandpass(x, fs, low=BANDPASS[0], high=BANDPASS[1], order=4):
    high = min(high, 0.45 * fs)
    sos = sps.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, x)


def notch(x, fs, f0=NOTCH_HZ, q=30.0):
    if f0 >= fs / 2:
        return x
    b, a = sps.iirnotch(f0, q, fs=fs)
    return sps.filtfilt(b, a, x)


def handle_noise(x, n_mad: float = 12.0):
    """Interpolate non-finite samples and clip gross artefacts (> n_mad*4 robust sigmas)."""
    x = np.asarray(x, dtype=float).copy()
    bad = ~np.isfinite(x)
    if bad.any():
        idx = np.arange(len(x))
        if (~bad).sum() < 2:
            return np.zeros_like(x)
        x[bad] = np.interp(idx[bad], idx[~bad], x[~bad])
    med = np.median(x)
    sigma = 1.4826 * np.median(np.abs(x - med))
    if sigma <= 0:
        return x
    lim = n_mad * sigma * 4
    return np.clip(x, med - lim, med + lim)


def preprocess(x, fs, powerline: bool = False, notch_hz: float = NOTCH_HZ) -> dict:
    """Single-channel pipeline; returns every stage for inspection."""
    x = np.asarray(x, dtype=float)
    nb = remove_baseline(handle_noise(x), fs)
    if powerline:
        nb = notch(nb, fs, notch_hz)
    return {"raw": x, "baseline_removed": nb, "filtered": bandpass(nb, fs)}


def normalise_by_r_amplitude(filtered, rpeaks):
    """Divide by median |R amplitude| (record-level) -> unit-independent amplitudes."""
    if len(rpeaks) == 0:
        s = np.std(filtered) or 1.0
    else:
        s = np.median(np.abs(filtered[rpeaks])) or np.std(filtered) or 1.0
    return filtered / s, float(s)
