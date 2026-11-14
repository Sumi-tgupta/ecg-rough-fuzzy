"""Evaluation helpers for the five-class AAMI heartbeat task."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def align_class_scores(scores: np.ndarray, model_classes, class_order=None) -> np.ndarray:
    """Place model score columns into a fixed class order, filling absent classes with zero."""
    class_order = list(class_order or config.CLASSES)
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("Model scores must be a two-dimensional array")
    if values.shape[1] != len(model_classes):
        raise ValueError("Number of score columns does not match model classes")
    aligned = np.zeros((values.shape[0], len(class_order)), dtype=float)
    for source_idx, label in enumerate(model_classes):
        if label in class_order:
            aligned[:, class_order.index(label)] = values[:, source_idx]
    return aligned


def evaluate_predictions(y_true, y_pred, scores=None, class_order=None) -> dict:
    """Return accuracy, balanced accuracy, macro/per-class metrics and OvR AUCs."""
    labels = list(class_order or config.CLASSES)
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred):
        raise ValueError("y_true and y_pred must be aligned one-dimensional arrays")
    if len(truth) == 0:
        raise ValueError("Cannot evaluate an empty test set")

    cm = confusion_matrix(truth, pred, labels=labels)
    precision = precision_score(truth, pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(truth, pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(truth, pred, labels=labels, average=None, zero_division=0)
    support = np.asarray([(truth == label).sum() for label in labels], dtype=int)
    observed_labels = [label for label, count in zip(labels, support) if count > 0]
    per_class = []
    for idx, label in enumerate(labels):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        specificity = tn / (tn + fp) if tn + fp else 0.0
        per_class.append({
            "class": label,
            "precision": float(precision[idx]),
            "recall_sensitivity": float(recall[idx]),
            "specificity": float(specificity),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        })

    metrics = {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_precision": float(precision_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(truth, pred, labels=observed_labels, average="macro", zero_division=0)),
        "macro_precision_all_aami": float(precision_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall_all_aami": float(recall_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1_all_aami": float(f1_score(truth, pred, labels=labels, average="macro", zero_division=0)),
        "macro_classes_evaluated": observed_labels,
        "weighted_f1": float(f1_score(truth, pred, labels=labels, average="weighted", zero_division=0)),
        "n_test": int(len(truth)),
        "class_order": labels,
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    if scores is not None:
        score_values = np.asarray(scores, dtype=float)
        if score_values.shape != (len(truth), len(labels)):
            raise ValueError("scores must have shape (n_samples, n_classes)")
        auc_rows = []
        for idx, label in enumerate(labels):
            binary = (truth == label).astype(int)
            if binary.min() == binary.max():
                continue
            auc_rows.append({
                "class": label,
                "roc_auc_ovr": float(roc_auc_score(binary, score_values[:, idx])),
                "average_precision_ovr": float(average_precision_score(binary, score_values[:, idx])),
            })
        metrics["ovr_auc"] = auc_rows
        metrics["macro_roc_auc"] = float(np.mean([r["roc_auc_ovr"] for r in auc_rows])) if auc_rows else None
        metrics["macro_average_precision"] = float(np.mean([r["average_precision_ovr"] for r in auc_rows])) if auc_rows else None
    else:
        metrics["ovr_auc"] = []
        metrics["macro_roc_auc"] = None
        metrics["macro_average_precision"] = None

    return metrics


def per_class_frame(model_name: str, metrics: dict) -> pd.DataFrame:
    return pd.DataFrame([{"model": model_name, **row} for row in metrics["per_class"]])
