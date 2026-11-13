# Changelog

## 0.2.0 — Phase 2: Research Engine Completion

- Added the Python fuzzy rule classifier with quantile memberships, beam-searched antecedents, coverage-based rule selection and sample-level explanations.
- Added reproducible record-wise feature collection and a four-model experiment pipeline.
- Added canonical Python feature schema and reusable AAMI metrics.
- Completed fuzzy-only and Rough-Set + fuzzy model wrappers, including model save/load support.
- Added record-overlap checks, training-only imputation and experiment manifests.
- Made `train.py` a functional CLI and replaced the incomplete Streamlit imports in `app.py` with a research console.
- Improved Rough Set handling of string labels, missing values and zero-dependency edge cases.
- Improved R-peak matching to enforce one-to-one reference/detection matches.
- Restricted the dominant-frequency feature to the documented 0.5–40 Hz band.
- Added deterministic unit and pipeline tests.
- Kept the browser's synthetic demonstration clearly separate from the MIT-BIH Python engine.

## 0.1.0 — Phase 1: Foundation and UI Refresh

- Introduced PulseLab branding and an Apple-inspired responsive browser interface.
- Added persistent light and dark appearance modes.
- Added design-system and architecture documentation.
- Added browser-core smoke test and Phase 1 notes.
