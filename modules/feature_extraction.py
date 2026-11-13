"""Beat-level feature extraction (time, frequency, statistical, wavelet, RR-interval).

Identical code is used for training records and uploaded signals so features are consistent.
Per-channel features are prefixed c0_, c1_ ... RR features are channel independent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
try:
    import pywt
except ImportError:  # dependency checked at wavelet-feature execution time
    pywt = None
from scipy import stats
from config import FS, get_logger
from modules.feature_schema import RR_FEATURES, CHANNEL_FEATURES as CH_FEATURES, feature_names, feature_domain
from modules import preprocessing as pp, rpeak_detection as rd, segmentation as sg

log = get_logger("features")

GROUPS = {"RR / rhythm": RR_FEATURES}


def rr_features(rpeaks: np.ndarray, fs: float) -> pd.DataFrame:
    rp = np.asarray(rpeaks, dtype=float)
    n = len(rp)
    rr = np.diff(rp) / fs if n > 1 else np.array([])
    med = np.median(rr) if len(rr) else 0.8
    prev = np.concatenate([[rr[0] if len(rr) else med], rr]) if n else np.array([])
    nxt = np.concatenate([rr, [rr[-1] if len(rr) else med]]) if n else np.array([])
    loc_mean, loc_std = np.empty(n), np.empty(n)
    for i in range(n):                      # local window of up to 10 RR intervals around the beat
        w = prev[max(i - 4, 0): i + 6]
        loc_mean[i], loc_std[i] = w.mean(), w.std()
    return pd.DataFrame({
        "rr_prev": prev, "rr_next": nxt, "rr_mean": loc_mean, "rr_std": loc_std,
        "rr_ratio_prev": prev / np.maximum(loc_mean, 1e-6), "rr_ratio_next": nxt / np.maximum(loc_mean, 1e-6),
        "heart_rate": 60.0 / np.maximum(loc_mean, 1e-6)})


def _qrs_duration(beats: np.ndarray, r0: int, fs: float) -> np.ndarray:
    """Width (s) of the region around R where |slope| exceeds 20% of the max |slope|."""
    d = np.abs(np.diff(beats, axis=1))
    w = int(0.14 * fs)
    lo, hi = max(r0 - w, 0), min(r0 + w, d.shape[1])
    seg = d[:, lo:hi]
    thr = 0.2 * seg.max(axis=1, keepdims=True)
    m = seg > thr
    first = m.argmax(axis=1)
    last = seg.shape[1] - 1 - m[:, ::-1].argmax(axis=1)
    return (last - first + 1) / fs


def channel_features(beats: np.ndarray, r0: int, fs: float) -> pd.DataFrame:
    n, L = beats.shape
    f = {}
    f["r_amp"] = beats[:, r0]
    f["qrs_dur"] = _qrs_duration(beats, r0, fs)
    f["energy"] = np.sum(beats ** 2, axis=1) / L
    f["variance"] = beats.var(axis=1)
    f["skewness"] = np.nan_to_num(stats.skew(beats, axis=1))
    f["kurtosis"] = np.nan_to_num(stats.kurtosis(beats, axis=1))
    f["ptp"] = np.ptp(beats, axis=1)
    f["area"] = np.sum(np.abs(beats), axis=1) / fs
    f["mean_pre_r"] = beats[:, : max(r0 - int(0.05 * fs), 1)].mean(axis=1)
    f["mean_post_r"] = beats[:, r0 + int(0.05 * fs):].mean(axis=1)
    t0, t1 = r0 + int(0.12 * fs), min(r0 + int(0.38 * fs), L)
    tseg = beats[:, t0:t1]
    f["t_wave_amp"] = tseg[np.arange(n), np.abs(tseg).argmax(axis=1)] if tseg.shape[1] else np.zeros(n)
    # frequency domain (zero-padded periodogram, Hann window)
    nfft = 512
    win = np.hanning(L)
    spec = np.abs(np.fft.rfft((beats - beats.mean(axis=1, keepdims=True)) * win, n=nfft, axis=1)) ** 2 / L
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    def band(a, b): return spec[:, (freqs >= a) & (freqs < b)].sum(axis=1)
    tot = spec[:, (freqs >= 0.5) & (freqs <= 40)].sum(axis=1) + 1e-12
    valid_freq = (freqs >= 0.5) & (freqs <= 40.0)
    f["dom_freq"] = freqs[valid_freq][spec[:, valid_freq].argmax(axis=1)]
    f["psd_0p5_5"], f["psd_5_15"], f["psd_15_40"] = band(0.5, 5) / tot, band(5, 15) / tot, band(15, 40) / tot
    p = spec[:, (freqs >= 0.5) & (freqs <= 40)] / tot[:, None]
    f["spec_entropy"] = -(p * np.log(p + 1e-12)).sum(axis=1)
    # wavelet (db4, level 4) relative energies
    if pywt is None:
        raise ImportError("PyWavelets is required for ECG wavelet features. Install it with: pip install PyWavelets")
    coeffs = pywt.wavedec(beats, "db4", level=4, axis=1)       # [cA4, cD4, cD3, cD2, cD1]
    en = np.stack([np.sum(c ** 2, axis=1) for c in coeffs], axis=1)
    en = en / (en.sum(axis=1, keepdims=True) + 1e-12)
    f["wav_a4"], f["wav_d4"], f["wav_d3"], f["wav_d2"] = en[:, 0], en[:, 1], en[:, 2], en[:, 3]
    return pd.DataFrame(f)[CH_FEATURES]


def process_signal(signal: np.ndarray, fs: float, rpeaks: np.ndarray | None = None,
                   powerline: bool = False, primary: int = 0) -> dict:
    """Full chain for a (n, channels) array. If `rpeaks` is given (annotations) they are refined on the
    primary channel; otherwise they are detected. Returns stages + a per-beat feature DataFrame."""
    sig = np.atleast_2d(np.asarray(signal, dtype=float))
    if sig.shape[0] < sig.shape[1]:
        sig = sig.T
    if sig.ndim != 2 or sig.shape[0] < 2 or sig.shape[1] < 1:
        raise ValueError("signal must contain at least two samples and one channel")
    primary = int(primary)
    if not 0 <= primary < sig.shape[1]:
        raise ValueError(f"primary channel index {primary} is outside {sig.shape[1]} available channels")
    if not np.isfinite(sig).any():
        raise ValueError("signal contains no finite samples")
    if fs != FS:
        sig = pp.resample_to(sig, fs, FS)
        if rpeaks is not None:
            rpeaks = np.round(np.asarray(rpeaks) * FS / fs).astype(int)
        fs = FS
    stages = [pp.preprocess(sig[:, c], fs, powerline) for c in range(sig.shape[1])]
    filt0 = stages[primary]["filtered"]
    rp = rd.detect_rpeaks(filt0, fs) if rpeaks is None else rd.refine_peaks(filt0, rpeaks, fs)
    norm, scales = [], []
    for st in stages:
        x, s = pp.normalise_by_r_amplitude(st["filtered"], rp)
        norm.append(x); scales.append(s)
    pre_n, post_n, _ = sg.beat_geometry(fs)
    beats, valid = sg.segment_beats(norm[primary], rp, fs)
    rp_valid = rp[valid]
    rrf = rr_features(rp, fs).loc[valid].reset_index(drop=True)      # RR from all peaks, then masked
    parts = [rrf]
    beats_all = []
    for c, x in enumerate(norm):
        b, _ = sg.segment_beats(x, rp, fs)
        beats_all.append(b)
        cf = channel_features(b, pre_n, fs)
        cf.columns = [f"c{c}_{k}" for k in cf.columns]
        parts.append(cf)
    feats = pd.concat(parts, axis=1)
    feats["r_sample"] = rp_valid
    return {"fs": fs, "stages": stages, "normalised": norm, "scales": scales, "rpeaks": rp,
            "valid": valid, "rpeaks_valid": rp_valid, "beats": beats_all, "features": feats,
            "n_channels": sig.shape[1], "signal": sig}


def build_record_features(record, primary: int = 0, max_channels: int = 2) -> pd.DataFrame:
    """Features for an MIT-BIH record using its annotated beat locations (both channels)."""
    sig = record.signal[:, :max_channels]
    out = process_signal(sig, record.fs, rpeaks=record.ann_samples, primary=primary)
    f = out["features"].copy()
    # map every refined R peak back to the nearest annotation (labels)
    ann, cls, sym = record.ann_samples, record.ann_classes, record.ann_symbols
    if len(ann) == 0:
        raise ValueError(f"Record {record.record_id} contains no AAMI-mappable beat annotations")
    ann_ds = np.round(ann * FS / record.fs).astype(int) if record.fs != FS else np.asarray(ann, dtype=int)
    beat_samples = f["r_sample"].to_numpy(dtype=int)
    # Assign each refined beat to its nearest reference annotation without assuming >= 2 annotations.
    if len(ann_ds) == 1:
        pick = np.zeros(len(beat_samples), dtype=int)
    else:
        order = np.searchsorted(ann_ds, beat_samples)
        right_idx = np.clip(order, 0, len(ann_ds) - 1)
        left_idx = np.clip(order - 1, 0, len(ann_ds) - 1)
        left_dist = np.abs(beat_samples - ann_ds[left_idx])
        right_dist = np.abs(ann_ds[right_idx] - beat_samples)
        pick = np.where(left_dist <= right_dist, left_idx, right_idx)
    ok = np.abs(ann_ds[pick] - beat_samples) <= int(0.1 * FS)
    f = f.loc[ok].reset_index(drop=True)
    pick = pick[ok]
    f.insert(0, "record", record.record_id)
    f["label"], f["symbol"] = cls[pick], sym[pick]
    return f
