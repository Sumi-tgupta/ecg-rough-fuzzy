"""Integrated Rough-Fuzzy model: Rough-Set reduct -> fuzzy rule base on selected features only."""
from __future__ import annotations
import json, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
import config
from modules.rough_set import RoughSetSelector
from modules.fuzzy_logic import FuzzyRuleClassifier


class FuzzyOnlyModel:
    """Model A: fuzzy classifier with ALL extracted features."""
    name = "A: Fuzzy (all features)"
    def __init__(self, **kw): self.fuzzy = FuzzyRuleClassifier(**kw)
    def fit(self, X, y):
        self.feature_names_ = list(X.columns); self.fuzzy.fit(X, y); self.classes_ = self.fuzzy.classes_; return self
    def predict(self, X): return self.fuzzy.predict(X[self.feature_names_])
    def scores(self, X): return self.fuzzy.normalised_scores(X[self.feature_names_])


class HybridRoughFuzzy:
    """Model B: Rough-Set selection (train only) followed by fuzzy classification on the reduct."""
    name = "B: Rough Set + Fuzzy"
    def __init__(self, n_bins=config.ROUGH_BINS, tolerance=config.ROUGH_TOLERANCE, **fuzzy_kw):
        self.rough = RoughSetSelector(n_bins, tolerance, config.ROUGH_MAX_PER_CLASS, config.SEED)
        self.fuzzy = FuzzyRuleClassifier(**fuzzy_kw)
        self.meta_ = {}

    def fit(self, X: pd.DataFrame, y):
        t0 = time.perf_counter()
        self.rough.fit(X, y)
        self.t_rough_ = time.perf_counter() - t0
        self.fuzzy.fit(X[self.rough.selected_], y)
        self.classes_ = self.fuzzy.classes_
        self.feature_names_ = list(X.columns)
        self.selected_ = self.rough.selected_
        return self

    def predict(self, X): return self.fuzzy.predict(X[self.selected_])
    def scores(self, X): return self.fuzzy.normalised_scores(X[self.selected_])
    def raw_scores(self, X): return self.fuzzy.decision_scores(X[self.selected_])
    def explain(self, row): return self.fuzzy.explain(row[self.selected_])

    def save(self, path: Path, extra_meta: dict | None = None):
        joblib.dump({"model": self, "meta": extra_meta or {}}, path)
        Path(path).with_suffix(".json").write_text(json.dumps({
            "selected_features": self.selected_, "classes": list(self.classes_),
            "rough_report": {k: v for k, v in self.rough.report().items() if k not in ("significance",)},
            "n_rules": len(self.fuzzy.rules_), "meta": extra_meta or {}}, indent=2, default=str))

    @staticmethod
    def load(path: Path):
        return joblib.load(path)["model"]
