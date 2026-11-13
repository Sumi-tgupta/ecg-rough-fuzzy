"""Conventional baselines on ALL features: Random Forest and RBF-SVM."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config


def _balanced_idx(y, cap, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    return np.concatenate([rng.choice(np.where(y == c)[0], min(cap, (y == c).sum()), replace=False) for c in np.unique(y)])


class RandomForestModel:
    name = "C1: Random Forest"
    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        self.m = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", n_jobs=-1,
                                        min_samples_leaf=2, random_state=config.SEED).fit(X.values, y)
        self.classes_ = list(self.m.classes_); return self
    def predict(self, X): return self.m.predict(X[self.feature_names_].values)
    def scores(self, X): return self.m.predict_proba(X[self.feature_names_].values)


class SVMModel:
    name = "C2: SVM (RBF)"
    def __init__(self, cap_per_class=4000): self.cap = cap_per_class
    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        idx = _balanced_idx(y, self.cap, config.SEED)        # class balancing on training data only
        self.m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=4.0, gamma="scale", class_weight="balanced",
                                                       decision_function_shape="ovr")).fit(X.values[idx], np.asarray(y)[idx])
        self.classes_ = list(self.m.classes_); return self
    def predict(self, X): return self.m.predict(X[self.feature_names_].values)
    def scores(self, X):                       # softmax over OvR decision values (a ranking score, not a probability)
        d = self.m.decision_function(X[self.feature_names_].values)
        if d.ndim == 1:                      # binary problem: sklearn returns one column
            d = np.column_stack([-d, d])
        e = np.exp(d - d.max(axis=1, keepdims=True)); return e / e.sum(axis=1, keepdims=True)
