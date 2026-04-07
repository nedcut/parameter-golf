"""Parameter Golf Lab — Experiment Dashboard.

Launch with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Parameter Golf Lab",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.ingest import load_all_runs  # noqa: E402
from dashboard.pages import comparison, overview, quantization, run_detail, size_budget, training_curves  # noqa: E402

# Page registry
PAGES = {
    "Overview": overview,
    "Training Curves": training_curves,
    "Quantization": quantization,
    "Comparison": comparison,
    "Size Budget": size_budget,
    "Run Detail": run_detail,
}

# Sidebar navigation
st.sidebar.title("⛳ Parameter Golf Lab")
if "page_name" not in st.session_state:
    st.session_state["page_name"] = "Overview"
page_name = st.sidebar.radio("Navigate", list(PAGES.keys()), key="page_name")

# Refresh button
if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()

# Load data
runs = load_all_runs()

# Global stats in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(runs)} runs loaded**")
complete = sum(1 for r in runs if r.status == "complete")
st.sidebar.markdown(f"{complete} complete, {len(runs) - complete} partial/crashed")

# Render selected page
PAGES[page_name].render(runs)
