"""Streamlit entrypoint: python -m streamlit run app.py"""
import streamlit as st
st.set_page_config(page_title="Rough-Fuzzy ECG Research Platform", page_icon="🫀", layout="wide")
from modules import ui
ui.inject()
pages = [st.Page("pages/overview.py", title="Overview", icon="🏠", default=True),
         st.Page("pages/signal_analysis.py", title="ECG Signal Analysis", icon="📈"),
         st.Page("pages/feature_analysis.py", title="Feature Extraction", icon="🧮"),
         st.Page("pages/rough_set_analysis.py", title="Rough Set Analysis", icon="🧩"),
         st.Page("pages/fuzzy_engine.py", title="Fuzzy Logic Engine", icon="🌫️"),
         st.Page("pages/prediction.py", title="Arrhythmia Prediction", icon="🫀"),
         st.Page("pages/model_evaluation.py", title="Model Evaluation", icon="📊"),
         st.Page("pages/research_report.py", title="Research Report", icon="📄")]
st.navigation(pages).run()
with st.sidebar:
    st.markdown("---"); st.caption("Academic research prototype. Not for clinical use.")
