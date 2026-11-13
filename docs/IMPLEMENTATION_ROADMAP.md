# Implementation Roadmap

## Phase 1 — Foundation and interface system

**Status: Implemented in this handoff.**

- Clarify current and target architecture.
- Add design tokens and responsive navigation shell.
- Add persistent light/dark themes.
- Preserve and accurately label the existing synthetic browser demonstration.
- Document current limitations and verification.

## Phase 2 — Research engine completion

**Status: Implemented in this handoff.** The Python MIT-BIH engine now has its own versioned feature schema; browser/Python schema consolidation is intentionally deferred to Phase 4.

- Implemented `modules/fuzzy_logic.py` with training-only membership fitting, rule induction, inference, scoring and explanations.
- Implemented `modules/pipeline.py` to orchestrate annotated record loading, feature assembly, train-only transformations, model fitting and experiment artifacts.
- Added class-balanced rough/fuzzy rule sampling, missing-class reporting and record / patient / subject overlap checks.
- Added reproducibility manifests with record splits, feature-schema version, model reports, metrics and configuration snapshots.
- Added CLI and Streamlit research-console entrypoints.
- Added deterministic unit and pipeline tests for fuzzy rules, Rough Set primitives, metrics, signal utilities and experiment artifact generation.

## Phase 3 — Real MIT-BIH experiments

- Install and pin dependencies; implement a controlled dataset download/cache flow.
- Verify AAMI annotation mapping and record exclusions.
- Test preprocessing and beat segmentation at dataset boundaries.
- Resolve channel policy and align Python/browser feature schemas (including RR descriptors and wavelet basis).
- Fit every learned transform on training records only.
- Evaluate on the established inter-patient split and report accuracy, macro-F1, balanced accuracy, per-class precision/recall/specificity/F1, confusion matrices and timing.
- Add confidence intervals or repeated patient-wise splits before making broad comparative claims.

## Phase 4 — Application integration

- Keep one UI and one authoritative research engine.
- Define API schemas for upload validation, prediction, feature inspection, model explanation, experiment listing and report export.
- Move long-running training into an explicit experiment workflow with progress and cancellation.
- Persist experiment metadata, model artifacts and reproducibility manifests.
- Make synthetic and real-data environments visually distinct.

## Phase 5 — Quality and research readiness

- Automated Python, JavaScript and API test suites.
- Input validation, file-size limits, sampling-rate/channel handling and safe error messages.
- Accessibility and browser compatibility checks.
- Performance profiling and large-signal testing.
- Reproducible setup, data provenance and model cards.
- Research documentation, limitations, ablation studies and final packaging.
