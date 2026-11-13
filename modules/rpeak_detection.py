"""Pan-Tompkins-style R-peak detector and annotation refinement."""
from __future__ import annotations
import numpy as np
from scipy import signal as sps


def detect_rpeaks(x, fs, refractory: float = 0.25) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sos = sps.butter(2, [5, min(15, 0.45 * fs)], btype="band", fs=fs, output="sos")
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
    """Sensitivity / PPV of detected R peaks against reference annotations."""
    tol_s = int(tol * fs)
    ref, det = np.asarray(reference), np.asarray(detected)
    if len(ref) == 0 or len(det) == 0:
        return {"tp": 0, "fn": len(ref), "fp": len(det), "sensitivity": 0.0, "ppv": 0.0}
    idx = np.searchsorted(det, ref)
    tp = sum(1 for r, i in zip(ref, idx)
             if np.min(np.abs(det[max(i - 1, 0):i + 1] - r)) <= tol_s)
    fp = max(len(det) - tp, 0)
    return {"tp": tp, "fn": len(ref) - tp, "fp": fp,
            "sensitivity": tp / len(ref), "ppv": tp / max(tp + fp, 1)}
