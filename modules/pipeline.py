"""Reproducible end-to-end MIT-BIH experiment pipeline for PulseLab."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

import config
from modules import baseline_models, data_loader, feature_extraction
from modules.feature_schema import FEATURE_SCHEMA_VERSION, feature_names as canonical_feature_names
from modules.hybrid_model import FuzzyOnlyModel, HybridRoughFuzzy
from modules.metrics import align_class_scores, evaluate_predictions, per_class_frame

Progress = Callable[[float, str], None] | None


def _notify(progress: Progress, fraction: float, message: str) -> None:
    if progress is not None:
        progress(float(np.clip(fraction, 0.0, 1.0)), message)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "model"


def _feature_columns(frame: pd.DataFrame, requested: Iterable[str] | None = None) -> list[str]:
    if requested is not None:
        columns = list(requested)
    else:
        reserved = {"record", "label", "symbol", "r_sample", "record_id", "patient_id"}
        columns = [c for c in frame.columns if c not in reserved and pd.api.types.is_numeric_dtype(frame[c])]
    if not columns:
        raise ValueError("No numeric feature columns were found")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature columns are missing from the table: {missing}")
    return columns


def _clean_feature_tables(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_columns: list[str]):
    for label, frame in (("training", train_df), ("testing", test_df)):
        if "label" not in frame.columns:
            raise ValueError(f"The {label} feature table must contain a 'label' column")
        if frame.empty:
            raise ValueError(f"The {label} feature table is empty")
    y_train = train_df["label"].astype(str).to_numpy()
    y_test = test_df["label"].astype(str).to_numpy()
    invalid = sorted((set(y_train) | set(y_test)) - set(config.CLASSES))
    if invalid:
        raise ValueError(f"Unexpected AAMI labels in feature tables: {invalid}")

    X_train = train_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_test = test_df.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    medians = X_train.median(axis=0).fillna(0.0)
    X_train = X_train.fillna(medians).fillna(0.0)
    X_test = X_test.fillna(medians).fillna(0.0)
    if not np.isfinite(X_train.to_numpy()).all() or not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Feature cleaning failed to produce finite numeric values")
    return X_train, y_train, X_test, y_test


def _model_suite(include_baselines: bool = True):
    models = [
        FuzzyOnlyModel(),
        HybridRoughFuzzy(),
    ]
    if include_baselines:
        models.extend([baseline_models.RandomForestModel(), baseline_models.SVMModel()])
    return models


def run_feature_experiment(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    *,
    feature_columns: Iterable[str] | None = None,
    output_root: Path | str | None = None,
    progress: Progress = None,
    include_baselines: bool = True,
    save_models: bool = True,
    run_name: str | None = None,
    metadata: dict | None = None,
    model_instances=None,
) -> Path:
    """Fit/evaluate the model suite on already-extracted, record-wise feature tables.

    This lower-level entry point is also used by tests, allowing the complete
    training/evaluation path to be exercised without downloading MIT-BIH data.
    """
    train_df = train_features.copy()
    test_df = test_features.copy()
    for identity_column, description in (("patient_id", "patient"), ("subject_id", "subject"), ("record", "record")):
        if identity_column in train_df.columns and identity_column in test_df.columns:
            train_ids = set(train_df[identity_column].astype(str))
            test_ids = set(test_df[identity_column].astype(str))
            overlap = sorted(train_ids & test_ids)
            if overlap:
                raise ValueError(f"{description.title()}-wise leakage detected; IDs occur in both splits: {overlap}")
    columns = _feature_columns(train_df, feature_columns)
    test_missing = [c for c in columns if c not in test_df.columns]
    if test_missing:
        raise ValueError(f"Testing feature table is missing training columns: {test_missing}")
    X_train, y_train, X_test, y_test = _clean_feature_tables(train_df, test_df, columns)

    run_id = run_name or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    if Path(run_id).name != run_id or run_id in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_name may contain only letters, numbers, dot, underscore and hyphen")
    root = Path(output_root) if output_root is not None else config.EXPERIMENTS_DIR
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_dir = config.MODELS_DIR / run_id
    if save_models:
        model_dir.mkdir(parents=True, exist_ok=False)

    all_models = list(model_instances) if model_instances is not None else _model_suite(include_baselines)
    results: dict[str, dict] = {}
    model_reports: dict[str, dict] = {}
    summary_rows = []
    per_class_tables = []
    score_kinds = {}
    n_models = len(all_models)

    for model_idx, model in enumerate(all_models):
        model_name = getattr(model, "name", model.__class__.__name__)
        slug = _safe_slug(model_name)
        start = time.perf_counter()
        _notify(progress, model_idx / max(n_models, 1), f"Training {model_name}")
        model.fit(X_train, y_train)
        train_seconds = time.perf_counter() - start

        predict_start = time.perf_counter()
        predictions = np.asarray(model.predict(X_test)).astype(str)
        predict_seconds = time.perf_counter() - predict_start
        score_matrix = None
        score_kind = "predicted labels only"
        if hasattr(model, "scores"):
            score_raw = model.scores(X_test)
            score_matrix = align_class_scores(score_raw, model.classes_, config.CLASSES)
            score_kind = "normalised fuzzy scores" if "Fuzzy" in model_name or "Rough" in model_name else "model ranking scores"
        elif hasattr(model, "predict_proba"):
            score_matrix = align_class_scores(model.predict_proba(X_test), model.classes_, config.CLASSES)
            score_kind = "predictive scores"

        metrics = evaluate_predictions(y_test, predictions, score_matrix, config.CLASSES)
        metrics["train_seconds"] = float(train_seconds)
        metrics["predict_seconds"] = float(predict_seconds)
        metrics["n_features_input"] = int(X_train.shape[1])
        metrics["n_features_used"] = int(len(getattr(model, "selected_", columns)))
        metrics["score_interpretation"] = score_kind
        results[model_name] = metrics
        score_kinds[model_name] = score_kind
        summary_rows.append({
            "model": model_name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_f1_all_aami": metrics["macro_f1_all_aami"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_roc_auc": metrics["macro_roc_auc"],
            "n_features_input": metrics["n_features_input"],
            "n_features_used": metrics["n_features_used"],
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
        })
        per_class_tables.append(per_class_frame(model_name, metrics))

        if hasattr(model, "fuzzy"):
            fuzzy_report = model.fuzzy.report()
            model_report = {"fuzzy": fuzzy_report}
            rule_payload = {"model": model_name, "fuzzy_report": fuzzy_report, "rules": model.fuzzy.rules_}
            if hasattr(model, "rough"):
                model_report["rough_set"] = model.rough.report()
                model_report["selected_features"] = list(model.selected_)
                model_report["removed_features"] = list(model.removed_)
                rule_payload["selected_features"] = list(model.selected_)
                rule_payload["rough_set_report"] = model.rough.report()
            model_reports[model_name] = model_report
            (run_dir / f"{slug}_rule_base.json").write_text(
                json.dumps(rule_payload, indent=2, default=_json_default), encoding="utf-8"
            )
        if hasattr(model, "m") and hasattr(model.m, "feature_importances_"):
            importance = pd.DataFrame({
                "feature": columns,
                "importance": model.m.feature_importances_,
            }).sort_values("importance", ascending=False)
            importance.to_csv(run_dir / f"{slug}_feature_importance.csv", index=False)
        if save_models:
            model_path = model_dir / f"{slug}.joblib"
            joblib.dump(model, model_path, compress=3)
            metrics["model_artifact"] = str(model_path.relative_to(config.ROOT)) if model_path.is_relative_to(config.ROOT) else str(model_path)
        cm_frame = pd.DataFrame(metrics["confusion_matrix"], index=config.CLASSES, columns=config.CLASSES)
        cm_frame.index.name = "actual\\predicted"
        cm_frame.to_csv(run_dir / f"{slug}_confusion_matrix.csv")
        _notify(progress, (model_idx + 1) / max(n_models, 1), f"Completed {model_name}")

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
    summary.to_csv(run_dir / "model_comparison.csv", index=False)
    pd.concat(per_class_tables, ignore_index=True).to_csv(run_dir / "per_class_metrics.csv", index=False)

    train_distribution = {str(c): int((y_train == c).sum()) for c in config.CLASSES}
    test_distribution = {str(c): int((y_test == c).sum()) for c in config.CLASSES}
    train_missing = [c for c in config.CLASSES if train_distribution[c] == 0]
    test_missing = [c for c in config.CLASSES if test_distribution[c] == 0]
    low_train_support = {c: count for c, count in train_distribution.items() if 0 < count < config.MIN_TRAIN_SAMPLES_PER_CLASS}
    config_snapshot = {
        "seed": config.SEED,
        "sampling_frequency_hz": config.FS,
        "bandpass_hz": list(config.BANDPASS),
        "baseline_median_windows_seconds": list(config.BASELINE_MEDIAN_WINDOWS),
        "notch_hz": config.NOTCH_HZ,
        "pre_r_seconds": config.PRE_R_SEC,
        "post_r_seconds": config.POST_R_SEC,
        "rough_bins": config.ROUGH_BINS,
        "rough_max_per_class": config.ROUGH_MAX_PER_CLASS,
        "rough_tolerance": config.ROUGH_TOLERANCE,
        "fuzzy_max_per_class": config.FUZZY_MAX_PER_CLASS,
        "fuzzy_rules_per_class": config.FUZZY_RULES_PER_CLASS,
        "fuzzy_max_antecedents": config.FUZZY_MAX_ANTECEDENTS,
        "fuzzy_beam_width": config.FUZZY_BEAM,
        "fuzzy_min_support": config.FUZZY_MIN_SUPPORT,
    }
    run_warnings = []
    if train_missing:
        run_warnings.append(f"No training examples for AAMI classes: {train_missing}")
    if test_missing:
        run_warnings.append(f"No testing examples for AAMI classes: {test_missing}; those classes cannot be evaluated on this split")
    if low_train_support:
        run_warnings.append(f"Low training support (below {config.MIN_TRAIN_SAMPLES_PER_CLASS} beats): {low_train_support}")
    run_manifest = {
        "project": "PulseLab",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": (metadata or {}).get("feature_schema_version", FEATURE_SCHEMA_VERSION),
        "feature_columns": columns,
        "n_features": len(columns),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_distribution": train_distribution,
        "test_class_distribution": test_distribution,
        "missing_train_classes": train_missing,
        "missing_test_classes": test_missing,
        "warnings": run_warnings,
        "configuration": config_snapshot,
        "train_records": sorted(map(str, set(train_df["record"]))) if "record" in train_df else [],
        "test_records": sorted(map(str, set(test_df["record"]))) if "record" in test_df else [],
        "models": list(results.keys()),
        "score_interpretation": score_kinds,
        "metrics": results,
        "model_reports": model_reports,
        "metadata": metadata or {},
        "disclaimer": config.DISCLAIMER,
        "notes": [
            "The test set must contain records/patients not present in training; overlap is rejected when record IDs are available.",
            "Fuzzy normalised scores and SVM ranking scores are not calibrated clinical probabilities.",
            "A single DS1/DS2 evaluation is not a confidence interval or clinical validation."
        ],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=_json_default), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    if save_models:
        (model_dir / "README.txt").write_text(
            "PulseLab research model artifacts. Load only trusted joblib files. "
            "These models are for academic research and are not clinically validated.\n",
            encoding="utf-8",
        )
    _notify(progress, 1.0, f"Experiment saved to {run_dir}")
    return run_dir


def collect_record_features(
    record_ids: Iterable[str],
    *,
    data_dir: Path | str = config.DATA_DIR,
    progress: Progress = None,
    record_loader: Callable = data_loader.load_record,
    feature_builder: Callable = feature_extraction.build_record_features,
) -> pd.DataFrame:
    """Load and extract features from each requested annotated MIT-BIH record."""
    ids = [str(r) for r in record_ids]
    if not ids:
        raise ValueError("At least one record ID is required")
    frames = []
    failures = []
    for idx, record_id in enumerate(ids):
        _notify(progress, idx / len(ids), f"Processing MIT-BIH record {record_id}")
        try:
            record = record_loader(record_id, Path(data_dir))
            frame = feature_builder(record)
            if frame is None or frame.empty:
                raise ValueError("feature extraction returned no heartbeat rows")
            if "record" not in frame:
                frame.insert(0, "record", record_id)
            if "label" not in frame:
                raise ValueError("feature table does not contain AAMI labels")
            frames.append(frame)
        except Exception as exc:
            failures.append(f"{record_id}: {type(exc).__name__}: {exc}")
        _notify(progress, (idx + 1) / len(ids), f"Finished record {record_id}")
    if failures:
        details = "\n".join(failures)
        raise RuntimeError(
            "Could not build a complete record feature table. No partial experiment was run.\n" + details
        )
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        raise RuntimeError("No annotated heartbeat features were extracted")
    return combined


def run_experiment(
    train_record_ids: Iterable[str] | None = None,
    test_record_ids: Iterable[str] | None = None,
    *,
    data_dir: Path | str = config.DATA_DIR,
    output_root: Path | str | None = None,
    progress: Progress = None,
    include_baselines: bool = True,
    save_models: bool = True,
    run_name: str | None = None,
) -> Path:
    """Extract record-wise MIT-BIH features and run the full model comparison."""
    train_ids = [str(r) for r in (train_record_ids or config.DS1)]
    test_ids = [str(r) for r in (test_record_ids or config.DS2)]
    if not train_ids or not test_ids:
        raise ValueError("Both training and testing record lists must be non-empty")
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Patient/record leakage detected before feature extraction: {overlap}")

    train = collect_record_features(
        train_ids, data_dir=data_dir,
        progress=lambda p, m: _notify(progress, 0.45 * p, "Training: " + m),
    )
    test = collect_record_features(
        test_ids, data_dir=data_dir,
        progress=lambda p, m: _notify(progress, 0.45 + 0.45 * p, "Testing: " + m),
    )
    feature_cols = [c for c in canonical_feature_names(2) if c in train.columns]
    if len(feature_cols) != len(canonical_feature_names(2)):
        missing = sorted(set(canonical_feature_names(2)) - set(feature_cols))
        raise ValueError(f"Extracted data does not match the two-channel feature schema; missing={missing}")
    metadata = {
        "dataset": "MIT-BIH Arrhythmia Database",
        "dataset_source": "PhysioNet WFDB records",
        "train_record_ids": train_ids,
        "test_record_ids": test_ids,
        "sampling_frequency_hz": config.FS,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "split_policy": "record-wise inter-patient DS1/DS2 split",
    }
    _notify(progress, 0.9, "Training and evaluating model suite")
    return run_feature_experiment(
        train, test,
        feature_columns=feature_cols,
        output_root=output_root,
        progress=lambda p, m: _notify(progress, 0.9 + 0.1 * p, m),
        include_baselines=include_baselines,
        save_models=save_models,
        run_name=run_name,
        metadata=metadata,
    )
