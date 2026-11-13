import numpy as np
import pandas as pd
import pytest

from modules.fuzzy_logic import FuzzyRuleClassifier


def make_data(seed=7):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for label, centre in [("N", -1.5), ("S", 0.0), ("V", 1.5)]:
        x = rng.normal(centre, 0.35, size=45)
        z = rng.normal(centre * 0.7, 0.4, size=45)
        rows.extend(zip(x, z))
        labels.extend([label] * len(x))
    return pd.DataFrame(rows, columns=["rr_ratio_prev", "c0_qrs_dur"]), np.asarray(labels)


def test_fuzzy_classifier_fit_predict_scores_and_explain():
    X, y = make_data()
    model = FuzzyRuleClassifier(
        n_rules_per_class=4, max_antecedents=2, beam_width=8,
        min_support=0.01, max_samples_per_class=100, seed=11,
    ).fit(X, y)

    pred = model.predict(X.iloc[:12])
    scores = model.normalised_scores(X.iloc[:12])
    assert pred.shape == (12,)
    assert scores.shape == (12, 3)
    assert np.all(np.isfinite(scores))
    assert np.allclose(scores.sum(axis=1), 1.0)
    assert model.n_rules_ > 0
    assert set(model.classes_) == {"N", "S", "V"}

    explanation = model.explain(X.iloc[0])
    assert explanation["prediction"] in {"N", "S", "V"}
    assert "memberships" in explanation
    assert isinstance(explanation["top_rules"], list)


def test_fuzzy_classifier_handles_missing_values_at_inference():
    X, y = make_data()
    model = FuzzyRuleClassifier(
        n_rules_per_class=2, max_antecedents=1, beam_width=5,
        min_support=0.0, max_samples_per_class=80,
    ).fit(X, y)
    sample = X.iloc[:3].copy()
    sample.iloc[0, 0] = np.nan
    sample.iloc[1, 1] = np.inf
    assert model.predict(sample).shape == (3,)


def test_fuzzy_classifier_rejects_missing_fitted_columns():
    X, y = make_data()
    model = FuzzyRuleClassifier(n_rules_per_class=1, max_antecedents=1, beam_width=3).fit(X, y)
    with pytest.raises(ValueError, match="missing fitted feature columns"):
        model.predict(X[["rr_ratio_prev"]])
