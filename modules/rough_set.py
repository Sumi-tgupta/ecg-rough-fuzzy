"""Rough Set Theory feature selection.

Implements: equal-frequency discretisation, indiscernibility relation, lower/upper approximations,
positive region, dependency degree gamma, attribute significance, core and greedy (QuickReduct-style)
reduct generation based on positive-region dependency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class Discretizer:
    """Equal-frequency discretisation; cuts and imputers are fitted on TRAIN data only."""

    def __init__(self, n_bins: int = 4):
        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2")
        self.n_bins = int(n_bins)
        self.cuts_, self.names_, self.medians_ = {}, [], None

    def fit(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if X.empty or X.shape[1] == 0:
            raise ValueError("Discretizer requires a non-empty feature matrix")
        if not X.columns.is_unique:
            raise ValueError("Feature names must be unique")
        try:
            numeric = X.apply(pd.to_numeric, errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Rough-set features must be numeric") from exc
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        self.names_ = list(numeric.columns)
        self.medians_ = numeric.median().fillna(0.0).to_numpy(dtype=float)
        values = numeric.fillna(pd.Series(self.medians_, index=self.names_)).to_numpy(dtype=float)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for idx, name in enumerate(self.names_):
            self.cuts_[name] = np.unique(np.quantile(values[:, idx], qs))
        self.n_features_in_ = len(self.names_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("Discretizer must be fitted before transform")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.names_)
        missing = [name for name in self.names_ if name not in X.columns]
        if missing:
            raise ValueError(f"Input is missing fitted discretiser columns: {missing}")
        numeric = X.loc[:, self.names_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = numeric.fillna(pd.Series(self.medians_, index=self.names_)).to_numpy(dtype=float)
        return np.column_stack([
            np.searchsorted(self.cuts_[name], values[:, idx], side="right")
            for idx, name in enumerate(self.names_)
        ]).astype(np.int64)


# ---------------------------------------------------------------- core RST primitives
def indiscernibility_classes(D: np.ndarray, attrs) -> np.ndarray:
    """Equivalence-class id for each object under IND(attrs).

    A single row-wise unique operation is substantially faster than repeatedly
    hashing one attribute at a time during QuickReduct's many subset checks.
    """
    D = np.asarray(D)
    if D.ndim != 2:
        raise ValueError("D must be a two-dimensional discretised decision table")
    attrs = list(attrs)
    if not attrs:
        return np.zeros(D.shape[0], dtype=np.int64)
    if min(attrs) < 0 or max(attrs) >= D.shape[1]:
        raise IndexError("An indiscernibility attribute index is outside D")
    _, gid = np.unique(D[:, attrs], axis=0, return_inverse=True)
    return gid.astype(np.int64, copy=False)


def _encode_decisions(y: np.ndarray, n_labels: int) -> np.ndarray:
    """Map arbitrary class labels to contiguous integer decision codes."""
    values = np.asarray(y)
    if values.ndim != 1:
        raise ValueError("Decision labels must be one-dimensional")
    _, encoded = np.unique(values, return_inverse=True)
    if len(np.unique(values)) > int(n_labels):
        raise ValueError("n_labels is smaller than the number of observed decision classes")
    return encoded.astype(np.int64)


def _class_label_table(gid: np.ndarray, y: np.ndarray, n_labels: int) -> np.ndarray:
    n_groups = gid.max() + 1
    return np.bincount(gid * n_labels + y, minlength=n_groups * n_labels).reshape(n_groups, n_labels)


def positive_region_mask(D, y, attrs, n_labels) -> np.ndarray:
    """Objects whose equivalence class lies wholly inside one decision class (union of lower approximations)."""
    y = _encode_decisions(y, n_labels)
    gid = indiscernibility_classes(D, attrs)
    tab = _class_label_table(gid, y, n_labels)
    consistent = (tab > 0).sum(axis=1) == 1
    return consistent[gid]


def dependency_degree(D, y, attrs, n_labels) -> float:
    """gamma_R(d) = |POS_R(d)| / |U|."""
    return float(positive_region_mask(D, y, attrs, n_labels).mean())


def approximations(D, y, attrs, n_labels) -> pd.DataFrame:
    """Lower/upper approximation sizes, boundary and accuracy of approximation per decision class."""
    y = _encode_decisions(y, n_labels)
    gid = indiscernibility_classes(D, attrs)
    tab = _class_label_table(gid, y, n_labels)
    sizes = tab.sum(axis=1)
    rows = []
    for k in range(n_labels):
        has_k = tab[:, k] > 0
        upper = int(sizes[has_k].sum())
        lower = int(sizes[has_k & ((tab > 0).sum(axis=1) == 1)].sum())
        rows.append({"class_index": k, "lower": lower, "upper": upper, "boundary": upper - lower,
                     "accuracy": lower / upper if upper else float("nan"), "n_objects": int((y == k).sum())})
    return pd.DataFrame(rows)


def attribute_significance(D, y, attrs, candidates, n_labels) -> dict:
    base = dependency_degree(D, y, attrs, n_labels)
    return {a: dependency_degree(D, y, list(attrs) + [a], n_labels) - base for a in candidates}


def core_attributes(D, y, n_labels, tol: float = 1e-12) -> list:
    """Attributes whose removal lowers gamma(C) (necessarily in every reduct)."""
    all_attrs = list(range(D.shape[1]))
    g_all = dependency_degree(D, y, all_attrs, n_labels)
    return [a for a in all_attrs
            if g_all - dependency_degree(D, y, [b for b in all_attrs if b != a], n_labels) > tol]


def quick_reduct(D, y, n_labels, tolerance: float = 0.0, max_attrs: int | None = None) -> dict:
    """Greedy positive-region reduct: repeatedly add the attribute with the largest gamma increase
    until gamma(R) >= gamma(C) - tolerance, then drop redundant attributes (backward elimination)."""
    y = _encode_decisions(y, n_labels)
    if D.ndim != 2 or D.shape[0] == 0 or D.shape[1] == 0:
        raise ValueError("QuickReduct requires a non-empty two-dimensional decision table")
    if len(y) != D.shape[0]:
        raise ValueError("Decision labels length does not match the decision table")
    n_attr = D.shape[1]
    g_full = dependency_degree(D, y, list(range(n_attr)), n_labels)
    reduct, history, g_cur = [], [], 0.0
    if g_full <= 1e-12:
        single = [dependency_degree(D, y, [a], n_labels) for a in range(n_attr)]
        best = int(np.argmax(single))
        reduct = [best]
        history.append({"step": 1, "added": best, "gamma_before": 0.0,
                        "gamma_after": float(single[best]), "significance": float(single[best])})
        return {"reduct": reduct, "gamma_full": g_full,
                "gamma_reduct": dependency_degree(D, y, reduct, n_labels), "history": history}
    while g_cur < g_full - tolerance - 1e-12 and len(reduct) < (max_attrs or n_attr):
        rest = [a for a in range(n_attr) if a not in reduct]
        gains = {a: dependency_degree(D, y, reduct + [a], n_labels) for a in rest}
        best = max(gains, key=gains.get)
        history.append({"step": len(history) + 1, "added": best, "gamma_before": g_cur,
                        "gamma_after": gains[best], "significance": gains[best] - g_cur})
        reduct.append(best)
        g_cur = gains[best]
    # backward elimination of redundant attributes (keeps gamma >= target)
    target = g_cur
    for a in list(reduct):
        trial = [b for b in reduct if b != a]
        if trial and dependency_degree(D, y, trial, n_labels) >= target - 1e-12:
            reduct = trial
    return {"reduct": reduct, "gamma_full": g_full, "gamma_reduct": dependency_degree(D, y, reduct, n_labels),
            "history": history}


class RoughSetSelector:
    def __init__(self, n_bins: int = 4, tolerance: float = 0.0, max_per_class: int = 2500, seed: int = 42):
        self.n_bins, self.tolerance, self.max_per_class, self.seed = n_bins, tolerance, max_per_class, seed

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        y = np.asarray(y)
        if X.empty or X.shape[1] == 0 or y.ndim != 1 or len(y) != len(X):
            raise ValueError("RoughSetSelector requires non-empty X and one label per row")
        if self.max_per_class < 1:
            raise ValueError("max_per_class must be positive")
        rng = np.random.default_rng(self.seed)
        self.feature_names_ = list(X.columns)
        self.classes_ = sorted(np.unique(y).tolist())
        yi = np.array([self.classes_.index(v) for v in y], dtype=np.int64)
        idx = np.concatenate([rng.choice(np.where(yi == k)[0], min(self.max_per_class, (yi == k).sum()), replace=False)
                              for k in range(len(self.classes_)) if (yi == k).sum() > 0])
        Xs, ys = X.iloc[idx], yi[idx]                    # class-balanced sample from training rows only
        self.disc_ = Discretizer(self.n_bins).fit(Xs)
        D = self.disc_.transform(Xs)
        n_lab = len(self.classes_)
        res = quick_reduct(D, ys, n_lab, self.tolerance)
        self.selected_idx_ = res["reduct"]
        self.selected_ = [self.feature_names_[i] for i in self.selected_idx_]
        self.removed_ = [f for f in self.feature_names_ if f not in self.selected_]
        self.gamma_full_, self.gamma_reduct_ = res["gamma_full"], res["gamma_reduct"]
        self.core_ = [self.feature_names_[i] for i in core_attributes(D, ys, n_lab)]
        self.history_ = [{**h, "feature": self.feature_names_[h["added"]]} for h in res["history"]]
        # significance of every attribute with respect to the full set and to the reduct
        all_idx = list(range(D.shape[1]))
        sig_full = {self.feature_names_[a]: self.gamma_full_ - dependency_degree(D, ys, [b for b in all_idx if b != a], n_lab)
                    for a in all_idx}
        sig_single = {self.feature_names_[a]: dependency_degree(D, ys, [a], n_lab) for a in all_idx}
        self.significance_ = pd.DataFrame({"feature": self.feature_names_,
            "gamma_single": [sig_single[f] for f in self.feature_names_],
            "sig_removal_from_full": [sig_full[f] for f in self.feature_names_],
            "in_reduct": [f in self.selected_ for f in self.feature_names_]}).sort_values(
            ["in_reduct", "gamma_single"], ascending=False).reset_index(drop=True)
        ap = approximations(D, ys, self.selected_idx_, n_lab)
        ap["class"] = [self.classes_[k] for k in ap["class_index"]]
        self.approximations_ = ap.drop(columns="class_index")
        self.approximations_full_ = approximations(D, ys, all_idx, n_lab).assign(
            **{"class": [self.classes_[k] for k in range(n_lab)]}).drop(columns="class_index")
        self.n_objects_ = len(ys)
        return self

    def report(self) -> dict:
        n, k = len(self.feature_names_), len(self.selected_)
        return {"n_original": n, "n_selected": k, "reduction_pct": 100.0 * (1 - k / n),
                "selected": self.selected_, "removed": self.removed_, "core": self.core_,
                "gamma_full": self.gamma_full_, "gamma_reduct": self.gamma_reduct_,
                "n_bins": self.n_bins, "n_objects": self.n_objects_, "history": self.history_,
                "significance": self.significance_.to_dict("records"),
                "approximations": self.approximations_.to_dict("records"),
                "approximations_full": self.approximations_full_.to_dict("records")}
