# PulseLab — Hybrid Rough-Fuzzy ECG Research Platform

PulseLab is an academic research prototype for ECG signal processing, Rough Set Theory-based feature reduction, and interpretable fuzzy-rule arrhythmia classification. The repository contains a polished browser demonstration and a separate Python research engine.

> **Research-use disclaimer:** PulseLab is not a medical device. Its outputs are not clinically validated and must not be used for diagnosis, treatment, triage, or clinical decision-making.

## Current release

**Version 0.2.0 — Phase 2: Research Engine Completion**

Phase 2 completes the Python fuzzy rule classifier and experiment pipeline, adds a canonical Python feature schema, provides a working CLI and Streamlit research console, and adds automated algorithm tests.

## 1. Run the Apple-inspired browser workspace

The static browser workspace needs no build tool or npm installation.

```powershell
# From the repository root on Windows
py -m http.server 8000
```

Open <http://localhost:8000/web/>.

The browser UI includes responsive layouts and persistent light/dark appearance. Its current models are trained on synthetic ECG generated in the browser. They are demonstration models, not MIT-BIH-trained models.

## 2. Set up the Python research engine

Use Python 3.10 or newer. On Windows PowerShell, from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The Python feature extraction path requires NumPy, SciPy, pandas and PyWavelets. MIT-BIH record loading and downloading use WFDB. The dataset is not included in the ZIP.

## 3. Run a quick engineering experiment

This downloads a smaller subset (8 training records and 6 testing records from the configured DS1 / DS2 lists), extracts ECG heartbeat features, and runs the model suite:

```powershell
py train.py --download --quick
```

The quick subset is intended to verify installation and execution. It is not the full research evaluation and may not contain examples from every AAMI class.

## 4. Run the complete configured DS1 / DS2 experiment

```powershell
py train.py --download
```

The experiment compares:

| Model | Description |
|---|---|
| A | Fuzzy classifier using all extracted features |
| B | Rough Set feature reduct followed by fuzzy classification |
| C1 | Random Forest baseline |
| C2 | RBF-SVM baseline |

To skip conventional baselines:

```powershell
py train.py --no-baselines
```

To download records separately, run `py train.py --download --quick` or `py train.py --download` once; subsequent runs use locally available records.

## 5. Open the Python research console

After dependencies are installed:

```powershell
streamlit run app.py
```

The console shows dataset availability, downloads the records needed for the chosen split, launches an experiment, and displays the saved model comparison and run metadata.

## 6. Run automated tests

```powershell
py -m pytest -q
node tests/browser_core_smoke.test.js
```

The Python tests use deterministic synthetic feature tables to exercise the fuzzy classifier, Rough Set selector, metrics and experiment artifact writer. They do not replace evaluation on the MIT-BIH test records.

## Outputs

Each run writes a timestamped folder under `experiments/`:

```text
experiments/<run_id>/
    run_manifest.json
    metrics.json
    model_comparison.csv
    per_class_metrics.csv
    *_confusion_matrix.csv
```

When model saving is enabled, model artifacts are written to `models/<run_id>/`. These local datasets and generated artifacts are ignored by Git by default.

Reported metrics include accuracy, balanced accuracy, macro/weighted F1, per-class precision, sensitivity/recall, specificity and (when mathematically defined) one-vs-rest ROC-AUC and average precision. Fuzzy and SVM scores are not presented as calibrated probabilities.

## Repository map

```text
config.py                    Shared experiment configuration and AAMI mapping
app.py                       Streamlit research console
train.py                     CLI for MIT-BIH experiments
modules/
  data_loader.py              WFDB record loading and AAMI labels
  preprocessing.py            Baseline removal, filtering, normalisation
  rpeak_detection.py          R-peak detection, refinement and matching
  segmentation.py             Beat-window extraction
  feature_schema.py            Canonical ordered Python feature schema
  feature_extraction.py        RR, morphology, statistical, spectral, wavelet features
  rough_set.py                 Discretisation, dependency, approximations and QuickReduct
  fuzzy_logic.py               Fuzzy membership functions, rules and explanations
  hybrid_model.py              Fuzzy-only and Rough-Set + fuzzy model wrappers
  baseline_models.py           Random Forest and RBF-SVM baselines
  metrics.py                   Fixed-order AAMI evaluation helpers
  pipeline.py                  Reproducible experiment orchestration and artifacts
web/
  index.html                   Browser application shell
  styles.css                   Phase 1 design system and responsive themes
  app.js                       Browser views and interactions
  core.js                      Synthetic-data browser demonstration algorithms
tests/                         Python and JavaScript automated tests
docs/                          Architecture, methodology and phase notes
```

## Research boundaries and interpretation

- The browser and Python feature pipelines are separate implementations. The browser currently uses a 25-feature synthetic schema; the Python MIT-BIH pipeline uses a 47-feature two-channel schema. They must not be treated as equivalent.
- The real-data Python pipeline must be run locally with downloaded MIT-BIH records. This repository does not bundle ECG data or model weights.
- All learned transformations are fitted on training data. The default DS1 / DS2 experiment uses disjoint record lists to reduce patient leakage.
- Quick-run metrics are engineering checks. Research conclusions should be based on the complete record-wise experiment and ideally additional independent splits.
- A single dataset split does not provide confidence intervals or clinical validation.

See `docs/PHASE_2_RESEARCH_ENGINE.md`, `docs/ARCHITECTURE.md` and `docs/METHODOLOGY.md` for implementation details.
