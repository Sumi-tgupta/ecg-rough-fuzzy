"""Model wrappers for fuzzy-only and Rough-Set + fuzzy experiments."""
from __future__ import annotations

import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

import config
from modules.fuzzy_logic import FuzzyRuleClassifier
from modules.rough_set import RoughSetSelector


class FuzzyOnlyModel:
    """Model A: fuzzy rule classifier trained on every extracted feature."""

    name = "A: Fuzzy (all features)"

    def __init__(self, **fuzzy_kwargs):
        self.fuzzy = FuzzyRuleClassifier(**fuzzy_kwargs)

    def fit(self, X: pd.DataFrame, y):
        self.feature_names_ = list(X.columns)
        self.fuzzy.fit(X, y)
        self.classes_ = self.fuzzy.classes_
        return self

    def predict(self, X):
        return self.fuzzy.predict(X[self.feature_names_])

    def scores(self, X):
        return self.fuzzy.normalised_scores(X[self.feature_names_])

    def raw_scores(self, X):
        return self.fuzzy.decision_scores(X[self.feature_names_])

    def explain(self, row):
        return self.fuzzy.explain(row[self.feature_names_])


class HybridRoughFuzzy:
    """Model B: training-only rough-set reduct followed by fuzzy classification."""

    name = "B: Rough Set + Fuzzy"

    def __init__(self, n_bins=config.ROUGH_BINS, tolerance=config.ROUGH_TOLERANCE, **fuzzy_kwargs):
        self.rough = RoughSetSelector(
            n_bins=n_bins,
            tolerance=tolerance,
            max_per_class=config.ROUGH_MAX_PER_CLASS,
            seed=config.SEED,
        )
        self.fuzzy = FuzzyRuleClassifier(**fuzzy_kwargs)
        self.meta_ = {}

    def fit(self, X: pd.DataFrame, y):
        started = time.perf_counter()
        self.feature_names_ = list(X.columns)
        self.rough.fit(X, y)
        self.t_rough_ = time.perf_counter() - started
        if not self.rough.selected_:
            raise RuntimeError("Rough-set selector returned an empty feature reduct")
        self.fuzzy.fit(X.loc[:, self.rough.selected_], y)
        self.classes_ = self.fuzzy.classes_
        self.selected_ = list(self.rough.selected_)
        self.removed_ = list(self.rough.removed_)
        return self

    def predict(self, X):
        return self.fuzzy.predict(X[self.selected_])

    def scores(self, X):
        return self.fuzzy.normalised_scores(X[self.selected_])

    def raw_scores(self, X):
        return self.fuzzy.decision_scores(X[self.selected_])

    def explain(self, row):
        return self.fuzzy.explain(row[self.selected_])

    def save(self, path: Path, extra_meta: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self, "meta": extra_meta or {}}, path, compress=3)
        report = {
            "selected_features": self.selected_,
            "removed_features": self.removed_,
            "classes": [str(c) for c in self.classes_],
            "rough_report": self.rough.report(),
            "fuzzy_report": self.fuzzy.report(),
            "n_rules": len(self.fuzzy.rules_),
            "meta": extra_meta or {},
        }
        path.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def load(path: Path):
        payload = joblib.load(path)
        return payload["model"] if isinstance(payload, dict) and "model" in payload else payload
