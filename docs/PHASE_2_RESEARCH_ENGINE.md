# Phase 2 — Research Engine Completion

**Release:** PulseLab 0.2.0  
**Scope:** Complete the missing Python fuzzy classifier and experiment pipeline, make the CLI and Streamlit research console usable, and add deterministic automated tests.

## What Phase 2 adds

- `modules/fuzzy_logic.py`: quantile-based Low / Medium / High membership functions, beam-searched rules of up to three distinct features, confidence-weighted coverage selection, product t-norm firing, probabilistic-OR class aggregation, normalised fuzzy scores, and per-sample rule explanations.
- `modules/pipeline.py`: record feature extraction, strict record-disjoint split validation, training-only feature imputation, four-model experiment orchestration, timing, fixed-order AAMI metrics, per-class sensitivity / specificity / F1, one-vs-rest ROC-AUC / average precision, model serialization, and run manifests.
- `modules/feature_schema.py`: canonical ordered Python feature schema and schema version.
- `modules/metrics.py`: reusable metrics and class-score alignment helpers.
- `modules/hybrid_model.py`: completed fuzzy-only and Rough-Set + fuzzy model wrappers, including save/load and explanations.
- `train.py`: usable command-line experiment runner with quick and full split modes.
- `app.py`: functional Streamlit research console for downloading records, launching experiments and inspecting saved results.
- `tests/`: deterministic checks for fuzzy rules, rough sets, metrics, the synthetic-feature experiment path, and the existing browser demo.

## Research safeguards

1. The Python pipeline rejects overlap between training and testing record IDs when the IDs are present in feature tables.
2. Feature medians, rough-set cut points, fuzzy membership quantiles and fuzzy rules are fitted using training rows only.
3. Test rows are transformed using the training-derived imputation values, rough-set reduct and fuzzy rules.
4. Model scores are labelled as fuzzy scores, predictive scores or ranking scores. They are not represented as calibrated clinical probabilities.
5. The five AAMI classes remain in a fixed evaluation order: N, S, V, F, Q. Classes absent in a quick subset are retained in the confusion matrix with zero support.
6. Artifacts carry dataset/split metadata and the project research disclaimer.

## Run locally on Windows

Install dependencies from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download and run a smaller engineering subset:

```powershell
py train.py --download --quick
```

Download and run the full configured DS1 / DS2 experiment:

```powershell
py train.py --download
```

Run the interactive Python research console after records are available:

```powershell
streamlit run app.py
```

Run automated checks:

```powershell
py -m pytest -q
node tests/browser_core_smoke.test.js
```

## Output structure

Each successful run creates:

```text
experiments/<run_id>/
    run_manifest.json
    metrics.json
    model_comparison.csv
    per_class_metrics.csv
    *_confusion_matrix.csv
    *_rule_base.json
    *_feature_importance.csv  (tree-based baseline when available)

models/<run_id>/
    a_fuzzy_all_features.joblib
    b_rough_set_fuzzy.joblib
    random_forest.joblib
    svm_rbf.joblib
```

The exact artifact list depends on selected model options. Model files are generated locally and are not committed to the repository by default.

## Current boundaries

- MIT-BIH records are not bundled. They are downloaded from PhysioNet on the user's machine using WFDB.
- The default quick subset is an engineering smoke run, not a substitute for the complete DS1 / DS2 research evaluation.
- The static Apple-inspired browser workspace continues to train its own synthetic demonstration models. Phase 2 does not silently connect it to the Python models; that integration belongs to a later phase.
- Full MIT-BIH execution requires installing the dependencies in `requirements.txt` and downloading the dataset. Environments without WFDB / PyWavelets cannot perform record extraction.
- PulseLab remains an academic research prototype, not a medical device. It has not been clinically validated.
