import json
import numpy as np
import pandas as pd
import pytest

from modules.pipeline import run_feature_experiment


def feature_tables():
    rng = np.random.default_rng(19)
    train_rows, test_rows = [], []
    centres = {"N": -1.0, "S": 0.0, "V": 1.0}
    for split, target in (("train", train_rows), ("test", test_rows)):
        for label, centre in centres.items():
            for idx in range(18):
                target.append({
                    "record": f"{split}_{label}_{idx // 6}",
                    "label": label,
                    "f1": rng.normal(centre, 0.25),
                    "f2": rng.normal(centre * 0.8, 0.3),
                    "f3": rng.normal(-centre, 0.35),
                })
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def test_pipeline_runs_python_fuzzy_and_hybrid_models(tmp_path):
    train, test = feature_tables()
    output = run_feature_experiment(
        train, test, feature_columns=["f1", "f2", "f3"],
        output_root=tmp_path, include_baselines=False, save_models=False,
        run_name="test-run",
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert set(manifest["models"]) == {"A: Fuzzy (all features)", "B: Rough Set + Fuzzy"}
    assert (output / "model_comparison.csv").exists()
    assert (output / "per_class_metrics.csv").exists()
    assert (output / "metrics.json").exists()
    assert (output / "a_fuzzy_all_features_rule_base.json").exists()
    assert (output / "b_rough_set_fuzzy_rule_base.json").exists()
    assert "model_reports" in manifest


def test_pipeline_rejects_record_overlap(tmp_path):
    train, test = feature_tables()
    test.loc[test.index[0], "record"] = train.loc[train.index[0], "record"]
    with pytest.raises(ValueError, match="Record-wise leakage"):
        run_feature_experiment(
            train, test, feature_columns=["f1", "f2", "f3"],
            output_root=tmp_path, include_baselines=False, save_models=False,
        )


def test_pipeline_can_train_all_four_models_and_report_baselines(tmp_path):
    train, test = feature_tables()
    output = run_feature_experiment(
        train, test, feature_columns=["f1", "f2", "f3"],
        output_root=tmp_path, include_baselines=True, save_models=False,
        run_name="four-model-check",
    )
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(manifest["models"]) == 4
    assert any("Random Forest" in name for name in manifest["models"])
    assert any("SVM" in name for name in manifest["models"])
    comparison = pd.read_csv(output / "model_comparison.csv")
    assert len(comparison) == 4
    assert "macro_f1_all_aami" in comparison.columns
