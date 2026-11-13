"""Heartbeat segmentation around R peaks."""
from __future__ import annotations
import numpy as np
from config import PRE_R_SEC, POST_R_SEC


def beat_geometry(fs, pre=PRE_R_SEC, post=POST_R_SEC):
    pre_n, post_n = int(round(pre * fs)), int(round(post * fs))
    return pre_n, post_n, pre_n + post_n


def segment_beats(x, rpeaks, fs, pre=PRE_R_SEC, post=POST_R_SEC):
    """Return (beats[n, L], valid_mask over rpeaks). R peak is at column pre_n."""
    pre_n, post_n, _ = beat_geometry(fs, pre, post)
    rp = np.asarray(rpeaks, dtype=int)
    valid = (rp - pre_n >= 0) & (rp + post_n <= len(x))
    idx = rp[valid][:, None] + np.arange(-pre_n, post_n)[None, :]
    return np.asarray(x)[idx], valid
