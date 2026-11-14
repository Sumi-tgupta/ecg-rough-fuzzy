import numpy as np
from modules.metrics import evaluate_predictions


def test_metrics_include_per_class_sensitivity_specificity_and_auc():
    y = np.array(["N", "N", "S", "S", "V", "V"])
    pred = np.array(["N", "S", "S", "S", "V", "N"])
    scores = np.array([
        [0.8, 0.1, 0.1, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.1, 0.8, 0.1, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.1, 0.1, 0.8, 0.0, 0.0],
        [0.5, 0.1, 0.4, 0.0, 0.0],
    ])
    result = evaluate_predictions(y, pred, scores)
    assert 0 <= result["accuracy"] <= 1
    assert result["per_class"][0]["specificity"] >= 0
    assert result["macro_roc_auc"] is not None
    assert len(result["confusion_matrix"]) == 5


def test_macro_f1_uses_test_present_classes_and_also_reports_all_aami_classes():
    y = np.array(["N", "N", "S", "S", "V", "V"])
    result = evaluate_predictions(y, y)
    assert result["macro_f1"] == 1.0
    assert result["macro_f1_all_aami"] == 0.6
    assert result["macro_classes_evaluated"] == ["N", "S", "V"]
