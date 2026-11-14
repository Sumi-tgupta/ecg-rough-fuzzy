"""Interpretable fuzzy rule classifier used by the Python research pipeline.

Membership parameters and rules are learned exclusively from the supplied
training data.  Inputs are pandas DataFrames so feature names remain attached
to every rule and explanation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import config


TERMS = ("low", "medium", "high")
_EPS = 1e-12


@dataclass(frozen=True)
class _Candidate:
    antecedents: tuple[tuple[int, int], ...]
    score: float
    support: float
    confidence: float


class FuzzyRuleClassifier:
    """Fuzzy classifier with quantile-based memberships and beam-searched rules.

    Parameters
    ----------
    n_rules_per_class:
        Maximum number of rules retained for each class.
    max_antecedents:
        Maximum number of distinct features in a rule.
    beam_width:
        Number of candidates retained at each beam-search depth.
    min_support:
        Minimum class-normalised fuzzy support for a candidate rule.
    max_samples_per_class:
        Maximum balanced training examples per class used for rule induction.
    seed:
        Random seed for reproducible balanced subsampling.
    """

    def __init__(
        self,
        n_rules_per_class: int = config.FUZZY_RULES_PER_CLASS,
        max_antecedents: int = config.FUZZY_MAX_ANTECEDENTS,
        beam_width: int = config.FUZZY_BEAM,
        min_support: float = config.FUZZY_MIN_SUPPORT,
        max_samples_per_class: int = config.FUZZY_MAX_PER_CLASS,
        seed: int = config.SEED,
    ):
        self.n_rules_per_class = int(n_rules_per_class)
        self.max_antecedents = int(max_antecedents)
        self.beam_width = int(beam_width)
        self.min_support = float(min_support)
        self.max_samples_per_class = int(max_samples_per_class)
        self.seed = int(seed)
        if self.n_rules_per_class < 1 or self.max_antecedents < 1 or self.beam_width < 1:
            raise ValueError("Rule count, antecedent count and beam width must be positive")
        if not 0 <= self.min_support <= 1:
            raise ValueError("min_support must be between 0 and 1")
        if self.max_samples_per_class < 1:
            raise ValueError("max_samples_per_class must be positive")

    @staticmethod
    def _as_frame(X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            arr = np.asarray(X)
            if arr.ndim != 2:
                raise ValueError("X must be a two-dimensional array or DataFrame")
            names = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]
            if len(names) != arr.shape[1]:
                raise ValueError("feature_names length does not match X")
            frame = pd.DataFrame(arr, columns=names)
        if frame.empty or frame.shape[1] == 0:
            raise ValueError("X must contain at least one row and one feature")
        if not frame.columns.is_unique:
            raise ValueError("Feature names must be unique")
        try:
            return frame.apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("All fuzzy classifier features must be numeric") from exc

    def _prepare_inference_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = self._as_frame(X)
        if isinstance(X, pd.DataFrame):
            missing = [name for name in self.feature_names_ if name not in frame.columns]
            if missing:
                raise ValueError(f"Input is missing fitted feature columns: {missing}")
            frame = frame.loc[:, self.feature_names_]
        elif frame.shape[1] != len(self.feature_names_):
            raise ValueError(f"Expected {len(self.feature_names_)} features, received {frame.shape[1]}")
        values = frame.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.medians_[None, :], values)
        return pd.DataFrame(values, columns=self.feature_names_, index=frame.index)

    def _membership_tensor(self, values: np.ndarray) -> np.ndarray:
        """Return n_samples x n_features x 3 Low/Medium/High memberships."""
        q15, q50, q85 = self.quantiles_.T
        low_span = np.maximum(q50 - q15, _EPS)
        high_span = np.maximum(q85 - q50, _EPS)
        low = np.where(values <= q15, 1.0, np.where(values < q50, (q50 - values) / low_span, 0.0))
        medium = np.maximum(0.0, np.minimum((values - q15) / low_span, (q85 - values) / high_span))
        high = np.where(values >= q85, 1.0, np.where(values > q50, (values - q50) / high_span, 0.0))
        return np.stack((low, medium, high), axis=2).clip(0.0, 1.0)

    @staticmethod
    def _activation(memberships: np.ndarray, antecedents: tuple[tuple[int, int], ...]) -> np.ndarray:
        activation = np.ones(memberships.shape[0], dtype=float)
        for feature_idx, term_idx in antecedents:
            activation *= memberships[:, feature_idx, term_idx]
        return activation

    @staticmethod
    def _score_activation(activation: np.ndarray, target_mask: np.ndarray) -> tuple[float, float, float]:
        target_sum = float(activation[target_mask].sum())
        support = target_sum / max(int(target_mask.sum()), 1)
        confidence = target_sum / max(float(activation.sum()), _EPS)
        score = 2.0 * support * confidence / max(support + confidence, _EPS)
        return score, support, confidence

    def _balanced_indices(self, y: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        selected = []
        for label in self.classes_:
            idx = np.flatnonzero(y == label)
            if len(idx) == 0:
                continue
            count = min(len(idx), self.max_samples_per_class)
            selected.append(rng.choice(idx, size=count, replace=False))
        if not selected:
            raise ValueError("No labelled training samples were supplied")
        return np.concatenate(selected)

    def _search_class_rules(self, memberships: np.ndarray, y: np.ndarray, label: Any) -> list[_Candidate]:
        target_mask = y == label
        n_samples, n_features, _ = memberships.shape
        beam: list[_Candidate] = []
        candidates_by_key: dict[tuple[tuple[int, int], ...], _Candidate] = {}

        # Depth one: each feature can contribute Low, Medium or High.
        for feature_idx in range(n_features):
            for term_idx in range(3):
                antecedents = ((feature_idx, term_idx),)
                activation = memberships[:, feature_idx, term_idx]
                score, support, confidence = self._score_activation(activation, target_mask)
                candidate = _Candidate(antecedents, score, support, confidence)
                candidates_by_key[antecedents] = candidate
        beam = sorted(candidates_by_key.values(), key=lambda c: (c.score, c.support, c.confidence), reverse=True)[: self.beam_width]

        for _depth in range(2, min(self.max_antecedents, n_features) + 1):
            expanded: dict[tuple[tuple[int, int], ...], _Candidate] = {}
            for parent in beam:
                used_features = {f for f, _ in parent.antecedents}
                for feature_idx in range(n_features):
                    if feature_idx in used_features:
                        continue
                    for term_idx in range(3):
                        antecedents = tuple(sorted(parent.antecedents + ((feature_idx, term_idx),)))
                        if antecedents in expanded:
                            continue
                        activation = self._activation(memberships, antecedents)
                        score, support, confidence = self._score_activation(activation, target_mask)
                        expanded[antecedents] = _Candidate(antecedents, score, support, confidence)
            if not expanded:
                break
            depth_beam = sorted(expanded.values(), key=lambda c: (c.score, c.support, c.confidence), reverse=True)[: self.beam_width]
            candidates_by_key.update({c.antecedents: c for c in depth_beam})
            beam = depth_beam

        candidates = list(candidates_by_key.values())
        eligible = [c for c in candidates if c.support >= self.min_support]
        if not eligible:
            eligible = candidates
        return sorted(eligible, key=lambda c: (c.score, c.support, c.confidence), reverse=True)

    def fit(self, X: pd.DataFrame | np.ndarray, y) -> "FuzzyRuleClassifier":
        frame = self._as_frame(X)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(frame):
            raise ValueError("y must be a one-dimensional array with one label per row")
        keep = pd.notna(labels)
        if not keep.all():
            frame = frame.loc[keep].reset_index(drop=True)
            labels = labels[keep]
        if len(labels) == 0:
            raise ValueError("No non-null training labels were supplied")

        self.feature_names_ = list(frame.columns)
        self.classes_ = list(np.unique(labels))
        values = frame.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        self.medians_ = np.nanmedian(values, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        values = np.where(np.isnan(values), self.medians_[None, :], values)
        self.n_features_in_ = values.shape[1]

        balanced_idx = self._balanced_indices(labels)
        balanced_values = values[balanced_idx]
        quantiles = np.quantile(balanced_values, [0.15, 0.50, 0.85], axis=0).T
        # Constant or heavily quantised features still need non-degenerate shoulders.
        spread = np.nanstd(balanced_values, axis=0)
        fallback = np.maximum(np.maximum(np.abs(quantiles[:, 1]) * 0.01, spread * 0.1), 1e-6)
        for j in range(quantiles.shape[0]):
            if quantiles[j, 0] >= quantiles[j, 1]:
                quantiles[j, 0] = quantiles[j, 1] - fallback[j]
            if quantiles[j, 2] <= quantiles[j, 1]:
                quantiles[j, 2] = quantiles[j, 1] + fallback[j]
        self.quantiles_ = quantiles

        balanced_y = labels[balanced_idx]
        memberships = self._membership_tensor(balanced_values)
        self.rules_: list[dict[str, Any]] = []
        self.rules_by_class_: dict[Any, list[dict[str, Any]]] = {}
        self.class_support_ = {label: int(np.sum(labels == label)) for label in self.classes_}

        for label in self.classes_:
            candidates = self._search_class_rules(memberships, balanced_y, label)
            target_mask = balanced_y == label
            current_coverage = np.zeros(int(target_mask.sum()), dtype=float)
            candidate_activations = [self._activation(memberships, c.antecedents)[target_mask] for c in candidates]
            available = set(range(len(candidates)))
            selected: list[dict[str, Any]] = []
            for _ in range(min(self.n_rules_per_class, len(candidates))):
                best_idx, best_utility, best_raw_gain = None, -1.0, 0.0
                for idx in available:
                    activation = candidate_activations[idx]
                    gain = float(np.maximum(activation - current_coverage, 0.0).mean())
                    # A confidence-weighted marginal coverage gain discourages vague rules.
                    utility = gain * max(candidates[idx].confidence, _EPS)
                    if utility > best_utility:
                        best_idx, best_utility, best_raw_gain = idx, utility, gain
                if best_idx is None:
                    break
                candidate = candidates[best_idx]
                antecedents = [
                    {"feature": self.feature_names_[f], "term": TERMS[t]}
                    for f, t in candidate.antecedents
                ]
                rule = {
                    "class": label,
                    "antecedents": antecedents,
                    "weight": float(np.clip(candidate.confidence, 0.0, 1.0)),
                    "support": float(candidate.support),
                    "confidence": float(candidate.confidence),
                    "score": float(candidate.score),
                    "coverage_gain": float(max(best_raw_gain, 0.0)),
                }
                selected.append(rule)
                current_coverage = np.maximum(current_coverage, candidate_activations[best_idx])
                available.remove(best_idx)
            self.rules_by_class_[label] = selected
            self.rules_.extend(selected)

        if not self.rules_:
            raise RuntimeError("Fuzzy rule induction produced no rules")
        self.n_rules_ = len(self.rules_)
        self.classes_ = np.asarray(self.classes_, dtype=object)
        return self

    def _fitted_memberships(self, X: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        if not hasattr(self, "rules_"):
            raise RuntimeError("FuzzyRuleClassifier must be fitted before prediction")
        frame = self._prepare_inference_frame(X)
        return frame, self._membership_tensor(frame.to_numpy(dtype=float))

    def decision_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Unnormalised class scores in [0, 1], aggregated by probabilistic OR."""
        _frame, memberships = self._fitted_memberships(X)
        scores = np.zeros((len(memberships), len(self.classes_)), dtype=float)
        for class_idx, label in enumerate(self.classes_):
            class_rules = self.rules_by_class_.get(label, [])
            if not class_rules:
                continue
            complement = np.ones(len(memberships), dtype=float)
            for rule in class_rules:
                ants = tuple((self.feature_names_.index(a["feature"]), TERMS.index(a["term"])) for a in rule["antecedents"])
                firing = self._activation(memberships, ants)
                complement *= 1.0 - np.clip(rule["weight"] * firing, 0.0, 1.0)
            scores[:, class_idx] = 1.0 - complement
        return scores

    def normalised_scores(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        scores = self.decision_scores(X)
        totals = scores.sum(axis=1, keepdims=True)
        zero_rows = totals[:, 0] <= _EPS
        result = scores / np.maximum(totals, _EPS)
        if zero_rows.any():
            result[zero_rows] = 1.0 / len(self.classes_)
        return result

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return row-normalised fuzzy class scores (not calibrated probabilities)."""
        return self.normalised_scores(X)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        scores = self.decision_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def explain(self, row: pd.Series | dict | pd.DataFrame, top_rules: int = 8) -> dict[str, Any]:
        """Explain one sample using feature memberships and strongest fired rules."""
        if isinstance(row, pd.Series):
            frame = row.to_frame().T
        elif isinstance(row, dict):
            frame = pd.DataFrame([row])
        elif isinstance(row, pd.DataFrame):
            if len(row) != 1:
                raise ValueError("explain expects exactly one row")
            frame = row
        else:
            raise TypeError("row must be a pandas Series, dict or one-row DataFrame")
        prepared, memberships = self._fitted_memberships(frame)
        scores = self.decision_scores(prepared)[0]
        normalised = scores / max(float(scores.sum()), _EPS)
        order = np.argsort(normalised)[::-1]
        values = memberships[0]
        membership_report = {
            name: {term.title(): float(values[i, j]) for j, term in enumerate(TERMS)}
            for i, name in enumerate(self.feature_names_)
        }
        fired = []
        for rule in self.rules_:
            ants = tuple((self.feature_names_.index(a["feature"]), TERMS.index(a["term"])) for a in rule["antecedents"])
            firing = float(self._activation(memberships, ants)[0])
            if firing > 0:
                fired.append({**rule, "firing_strength": firing, "contribution": float(rule["weight"] * firing)})
        fired.sort(key=lambda r: r["contribution"], reverse=True)
        best = int(order[0])
        margin = float(normalised[order[0]] - normalised[order[1]]) if len(order) > 1 else float(normalised[order[0]])
        return {
            "prediction": self.classes_[best],
            "class_scores": {str(label): float(normalised[i]) for i, label in enumerate(self.classes_)},
            "raw_scores": {str(label): float(scores[i]) for i, label in enumerate(self.classes_)},
            "confidence_score": float(normalised[best]),
            "top_two_margin": margin,
            "active_rule_count": sum(r["firing_strength"] > 0 for r in fired),
            "no_rule_fired": len(fired) == 0,
            "memberships": membership_report,
            "top_rules": fired[: max(int(top_rules), 0)],
        }

    def report(self) -> dict[str, Any]:
        if not hasattr(self, "rules_"):
            raise RuntimeError("FuzzyRuleClassifier must be fitted before report")
        return {
            "n_features": len(self.feature_names_),
            "n_classes": len(self.classes_),
            "n_rules": self.n_rules_,
            "rules_per_class": {str(k): len(v) for k, v in self.rules_by_class_.items()},
            "classes": [str(c) for c in self.classes_],
            "feature_names": list(self.feature_names_),
            "membership_quantiles": self.quantiles_.tolist(),
            "class_support": {str(k): int(v) for k, v in self.class_support_.items()},
        }
