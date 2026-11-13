"""Streamlit research console for the PulseLab Python experiment engine."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from modules import data_loader, pipeline

st.set_page_config(page_title="PulseLab Research Console", page_icon="📈", layout="wide")

st.title("PulseLab Research Console")
st.caption("Hybrid Rough-Fuzzy ECG Research Platform · Python experiment engine")
st.warning(config.DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("Experiment setup")
    run_mode = st.radio("Record split", ["Quick engineering run", "Full DS1 / DS2 evaluation"], index=0)
    include_baselines = st.checkbox("Include Random Forest and RBF-SVM", value=True)
    save_models = st.checkbox("Save trained model artifacts", value=True)
    st.divider()
    st.caption(f"Dataset directory: `{config.DATA_DIR}`")

train_ids = config.DS1[:8] if run_mode == "Quick engineering run" else config.DS1
test_ids = config.DS2[:6] if run_mode == "Quick engineering run" else config.DS2

available = set(data_loader.available_records(config.DATA_DIR))
requested = list(train_ids) + list(test_ids)
missing = [rid for rid in requested if rid not in available]

col1, col2, col3 = st.columns(3)
col1.metric("Records available", len(available))
col2.metric("Records required", len(requested))
col3.metric("Missing records", len(missing))

if missing:
    st.info(
        "The selected MIT-BIH records are not all present locally. Download them before running an experiment. "
        "The dataset is not bundled with PulseLab."
    )
    if st.button("Download selected MIT-BIH records", type="primary"):
        bar = st.progress(0.0)
        status = st.empty()
        try:
            data_loader.download_records(
                requested,
                progress=lambda fraction, message: (bar.progress(fraction), status.write(message)),
            )
            st.success("Download step finished. Refresh the page to verify the local records.")
            st.rerun()
        except Exception as exc:
            st.error(f"Dataset download failed: {type(exc).__name__}: {exc}")
else:
    st.success("All records required for the selected split are available locally.")
    if st.button("Run research experiment", type="primary"):
        bar = st.progress(0.0)
        status = st.empty()
        try:
            result_path = pipeline.run_experiment(
                train_ids,
                test_ids,
                progress=lambda fraction, message: (bar.progress(fraction), status.write(message)),
                include_baselines=include_baselines,
                save_models=save_models,
            )
            st.session_state["last_run_path"] = str(result_path)
            st.success(f"Experiment completed: {result_path}")
        except Exception as exc:
            st.error(f"Experiment failed: {type(exc).__name__}: {exc}")

last_run = st.session_state.get("last_run_path")
if last_run and Path(last_run, "run_manifest.json").exists():
    st.divider()
    st.subheader("Latest experiment")
    manifest = json.loads(Path(last_run, "run_manifest.json").read_text(encoding="utf-8"))
    st.caption(f"Run ID: {manifest['run_id']} · Created: {manifest['created_at_utc']}")
    comparison_path = Path(last_run, "model_comparison.csv")
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        metric_col = "macro_f1"
        if metric_col in comparison.columns:
            st.bar_chart(comparison.set_index("model")[[metric_col]])
    with st.expander("Run manifest and research limitations"):
        st.json({k: v for k, v in manifest.items() if k not in {"metrics"}})
        st.json(manifest.get("metrics", {}))

st.divider()
st.caption("The static browser workspace is available separately at `http://localhost:8000/web/` when served from the repository root.")
