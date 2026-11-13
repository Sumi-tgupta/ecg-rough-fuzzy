"""Canonical Python feature schema for PulseLab's MIT-BIH research pipeline.

The browser demo intentionally retains its separate synthetic 25-feature schema;
this module is the source of truth for the real-data Python pipeline.
"""
from __future__ import annotations

FEATURE_SCHEMA_VERSION = "python-mitdb-1.0.0"

RR_FEATURES = [
    "rr_prev", "rr_next", "rr_mean", "rr_std",
    "rr_ratio_prev", "rr_ratio_next", "heart_rate",
]

CHANNEL_FEATURES = [
    "r_amp", "qrs_dur", "energy", "variance", "skewness", "kurtosis",
    "ptp", "area", "mean_pre_r", "mean_post_r", "t_wave_amp", "dom_freq",
    "psd_0p5_5", "psd_5_15", "psd_15_40", "spec_entropy",
    "wav_a4", "wav_d4", "wav_d3", "wav_d2",
]


def feature_names(n_channels: int) -> list[str]:
    """Return ordered feature names for the requested number of ECG channels."""
    if not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer")
    return RR_FEATURES + [
        f"c{channel}_{name}"
        for channel in range(n_channels)
        for name in CHANNEL_FEATURES
    ]


def feature_domain(name: str) -> str:
    """Map a canonical feature name to its human-readable research domain."""
    if name in RR_FEATURES:
        return "RR / rhythm"
    base = name.split("_", 1)[1] if name.startswith("c") and len(name) > 1 and name[1].isdigit() else name
    if base in {"psd_0p5_5", "psd_5_15", "psd_15_40", "dom_freq", "spec_entropy"}:
        return "Frequency"
    if base.startswith("wav_"):
        return "Wavelet"
    if base in {"variance", "skewness", "kurtosis"}:
        return "Statistical"
    return "Time / morphology"


def validate_feature_columns(columns, n_channels: int | None = None) -> list[str]:
    """Validate feature-column ordering and return it as a list."""
    result = list(columns)
    if not result or len(result) != len(set(result)):
        raise ValueError("Feature columns must be non-empty and unique")
    if n_channels is not None:
        expected = feature_names(n_channels)
        missing = [name for name in expected if name not in result]
        extra = [name for name in result if name not in expected]
        if missing or extra:
            raise ValueError(f"Feature schema mismatch; missing={missing}, extra={extra}")
        return expected
    return result
