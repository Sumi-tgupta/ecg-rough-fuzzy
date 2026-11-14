import numpy as np

from modules.feature_extraction import rr_features
from modules.rpeak_detection import match_detections
from modules.segmentation import segment_beats
from modules.feature_schema import feature_names


def test_rpeak_matching_is_one_to_one():
    result = match_detections([100, 105, 300], [102, 302], fs=360, tol=0.02)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0
    assert len(result["matched_pairs"]) == 2


def test_segment_beats_returns_fixed_length_windows_and_valid_mask():
    signal = np.arange(1000, dtype=float)
    beats, valid = segment_beats(signal, [9, 500, 995], fs=100, pre=0.1, post=0.2)
    assert valid.tolist() == [False, True, False]
    assert beats.shape == (1, 30)
    assert beats[0, 10] == 500


def test_python_feature_schema_is_47_columns_for_two_channels():
    names = feature_names(2)
    assert len(names) == 47
    assert len(names) == len(set(names))
    assert names[:7] == ["rr_prev", "rr_next", "rr_mean", "rr_std", "rr_ratio_prev", "rr_ratio_next", "heart_rate"]


def test_rr_features_are_finite_and_aligned_to_rpeaks():
    result = rr_features(np.array([100, 460, 820, 1180]), fs=360)
    assert len(result) == 4
    assert np.isfinite(result.to_numpy()).all()
