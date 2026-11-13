# System Architecture

## 1. Two execution paths

PulseLab currently contains a browser demonstration and a Python research engine. They are intentionally separate in this release.

### Browser demonstration

```text
web/index.html
   ├── web/styles.css       Apple-inspired design system, responsive layout, themes
   ├── web/app.js           Navigation, views, interactions, canvas rendering
   └── web/core.js          Synthetic ECG generator, processing, features,
                            rough-set selector, fuzzy rules, evaluation
```

The browser application is static and client-side. It generates synthetic ECG records when the page loads and trains its demonstration models in the browser. It does not call the Python modules, load MIT-BIH records, or use Python model artifacts.

### Python research engine

```text
MIT-BIH / WFDB records
        ↓
modules.data_loader
        ↓
modules.preprocessing
        ↓
modules.rpeak_detection
        ↓
modules.segmentation
        ↓
modules.feature_extraction + feature_schema
        ↓
        ┌─────────────────────────────────────────┐
        │ Model A: fuzzy rules on all features    │
        │ Model B: Rough Set reduct → fuzzy rules │
        │ Model C1: Random Forest                 │
        │ Model C2: RBF-SVM                       │
        └─────────────────────────────────────────┘
        ↓
modules.metrics + pipeline
        ↓
experiments/<run_id>/ + models/<run_id>/
```

`train.py` is the CLI entrypoint. `app.py` is a Streamlit research console. Both call the same `modules.pipeline` implementation.

## 2. Research engine contracts

- `modules.feature_schema` is the canonical ordered schema for Python MIT-BIH features. The current two-channel schema contains 7 RR features plus 20 features per channel (47 total).
- The browser's 25-feature synthetic schema remains separate until explicit integration work; similarly named features must not be assumed to be interchangeable.
- Preprocessing, R-peak locations, segmentation and feature extraction use the same Python functions for train and inference preparation.
- Training-only transformations include missing-value medians, rough-set discretisation cuts and fuzzy membership quantiles.
- The pipeline validates that record IDs do not overlap between train and test feature tables.
- Evaluation uses the fixed AAMI order N, S, V, F, Q and reports per-class metrics as well as aggregate scores.
- Every saved experiment includes a run manifest with split, schema, model, metrics and research limitations.

## 3. Python fuzzy classifier

The Python fuzzy model fits a three-term membership partition per feature using training-data quantiles (15th, 50th and 85th percentiles). It induces class-specific conjunction rules with a bounded beam search. Rule antecedents use distinct features and are limited to three by default. Rule selection uses a confidence-weighted marginal coverage criterion. At inference, antecedent memberships are combined using product t-norm; class rules are aggregated using probabilistic OR.

The resulting row-normalised fuzzy scores are useful for ranking and explanation, but are not calibrated probabilities.

## 4. Rough Set selector

The selector learns equal-frequency bin boundaries from a balanced training-only sample, computes indiscernibility classes and positive-region dependency, then constructs a greedy QuickReduct-style subset with backward pruning. It reports the full/reduct dependency, core, selected/removed attributes and approximation sizes.

The reduct is consistency-oriented, not directly optimised for predictive accuracy. The experiment therefore compares the hybrid model against the fuzzy-only model and conventional baselines.

## 5. Data and split policy

The default evaluation uses the configured inter-patient DS1 / DS2 record lists. All beats from a record belong to exactly one side of the split. The dataset is downloaded locally from PhysioNet using WFDB and is not committed or bundled in this repository.

## 6. Non-clinical status

PulseLab is an academic research prototype. It is not a medical device, and its predictions must not be used for diagnosis, triage, treatment or clinical decisions.
