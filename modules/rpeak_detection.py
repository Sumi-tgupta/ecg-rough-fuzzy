"""Pan-Tompkins-style R-peak detector and annotation refinement."""
from __future__ import annotations
import numpy as np
from scipy import signal as sps


def detect_rpeaks(x, fs, refractory: float = 0.25) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 8:
        raise ValueError("R-peak detection requires a one-dimensional signal with at least 8 samples")
    if fs <= 0 or refractory <= 0:
        raise ValueError("fs and refractory must be positive")
    high = min(15.0, 0.45 * fs)
    if high <= 5.0:
        raise ValueError("Sampling frequency is too low for the 5-15 Hz detector band")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    sos = sps.butter(2, [5, high], btype="band", fs=fs, output="sos")
    f = sps.sosfiltfilt(sos, x)
    sq = np.gradient(f) ** 2
    win = max(int(0.12 * fs), 1)
    integ = np.convolve(sq, np.ones(win) / win, mode="same")
    thr = 0.3 * np.percentile(integ, 99.5) if np.any(integ) else 0
    peaks, _ = sps.find_peaks(integ, height=thr, distance=max(int(refractory * fs), 1))
    return refine_peaks(x, peaks, fs)


def refine_peaks(x, peaks, fs, half_window: float = 0.05) -> np.ndarray:
    """Snap candidates to the sample of largest |amplitude| within +/- half_window s."""
    hw = max(int(half_window * fs), 1)
    out = []
    for p in np.asarray(peaks, dtype=int):
        lo, hi = max(p - hw, 0), min(p + hw + 1, len(x))
        if hi > lo:
            out.append(lo + int(np.argmax(np.abs(x[lo:hi]))))
    return np.unique(np.array(out, dtype=int))


def match_detections(detected, reference, fs, tol: float = 0.075) -> dict:
    """One-to-one sensitivity / PPV matching against reference annotations.

    A detected peak and a reference annotation can contribute to at most one
    true positive.  This avoids over-counting when detections cluster around
    the same reference beat.
    """
    if fs <= 0 or tol < 0:
        raise ValueError("fs must be positive and tol must be non-negative")
    tol_samples = int(round(tol * fs))
    ref = np.sort(np.asarray(reference, dtype=int))
    det = np.sort(np.asarray(detected, dtype=int))
    i = j = tp = 0
    matched = []
    while i < len(ref) and j < len(det):
        delta = int(det[j] - ref[i])
        if abs(delta) <= tol_samples:
            tp += 1
            matched.append((int(ref[i]), int(det[j])))
            i += 1
            j += 1
        elif delta < -tol_samples:
            j += 1
        else:
            i += 1
    fn = len(ref) - tp
    fp = len(det) - tp
    return {
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "sensitivity": float(tp / len(ref)) if len(ref) else 0.0,
        "ppv": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "matched_pairs": matched,
    }
