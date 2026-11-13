# Architecture

```
ECG (WFDB / CSV) -> data_loader -> preprocessing -> rpeak_detection -> segmentation -> feature_extraction
      -> [train only] rough_set (discretise -> IND -> POS -> gamma -> reduct)
      -> fuzzy_logic (fuzzify -> rule firing -> aggregation -> decision) -> hybrid_model
      -> evaluation (A fuzzy-all, B rough+fuzzy, C RF/SVM) -> experiments/<run>/ + models/*.joblib
      -> Streamlit pages (read stored experiment; inspect every intermediate quantity)
```
* **pipeline.py** orchestrates a reproducible run (seed in `config.py`), writes `results.json`, CSVs, feature tables and the hybrid model with a JSON sidecar listing the reduct and rule count.
* **service.py** holds cached loaders for the UI; **ui.py** the CSS theme; **visualization.py** all Plotly figures; **report.py** CSV/PDF export.
* The same `feature_extraction.process_signal` is used for training and uploaded signals (consistency requirement). Uploads are resampled to 360 Hz.
* Persistence: `models/hybrid_rough_fuzzy.joblib` (full object: discretiser, reduct, membership centres, rules). Baselines are re-fit per run and not persisted.
* Logging via `config.get_logger`; invalid input raises `RecordError` / `ValueError` and is surfaced in the UI.
