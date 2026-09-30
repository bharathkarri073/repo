# modules/sidebar_ui.py
"""
Sidebar UI components for the GenAI Data Analyst platform.

Public API is unchanged: render_sidebar_brand(), render_workflow_status(),
render_sidebar_footer() are still the three entry points streamlit_app.py
calls. Internals now use the theme classes from custom_css.py
(.status-pill, .nav-section-label) instead of ad-hoc inline styling, and
the pipeline status widget gains dataset/model metadata so it reads like
a real ops panel instead of a checklist.
"""

import streamlit as st
from textwrap import dedent


def render_sidebar_brand():
    """
    Draws the branded header at the top of the Streamlit sidebar.
    """

    st.sidebar.markdown("### 🤖")
    st.sidebar.markdown("#### GenAI Analyst")
    st.sidebar.caption("Autonomous Analytics Platform")
    st.sidebar.success("AI SYSTEM ONLINE")


def _pipeline_steps():
    """
    Builds the ordered pipeline step list from session state.
    Kept as its own function so both the sidebar widget and any future
    dashboard "pipeline progress" component can share the same source
    of truth instead of duplicating the session-state lookups.
    """

    preprocessing_done = "preprocessing_done" in st.session_state
    engineering_done = "engineered_df" in st.session_state
    automl_done = "automl_results" in st.session_state
    llm_done = bool(st.session_state.get("report"))

    return [
        ("Data loaded", "df" in st.session_state),
        ("Preprocessing", preprocessing_done),
        ("Feature engineering", engineering_done),
        ("AutoML training", automl_done),
        ("AI report", llm_done),
    ]


def render_workflow_status():
    """
    Shows project pipeline status in the sidebar using session state.
    """

    steps = _pipeline_steps()
    completed_count = sum(1 for _, done in steps if done)
    total_count = len(steps)

    st.sidebar.markdown("### ⚡ Pipeline Status")

    progress_pct = int((completed_count / total_count) * 100) if total_count else 0
    st.sidebar.progress(progress_pct / 100)
    st.sidebar.caption(f"{completed_count} / {total_count} stages complete")

    for title, completed in steps:
        icon = "✅" if completed else "◯"
        status_label = "DONE" if completed else "PENDING"
        label = f"{icon} {title}"
        st.sidebar.markdown(f"**{label}**")
        st.sidebar.caption(status_label)

    _render_dataset_status()

    st.sidebar.markdown("---")


def _render_dataset_status():
    """
    Small metadata strip showing dataset shape and detected target/problem
    type, when available, so the sidebar communicates more than a bare
    checklist while a session is active.
    """

    df = st.session_state.get("df")

    if df is None or df.empty:
        return

    rows, cols = df.shape

    st.sidebar.info(f"Active dataset\n{rows:,} rows · {cols} columns")


def render_sidebar_footer():
    st.sidebar.markdown("### Technology Stack")
    st.sidebar.caption("Python · Streamlit · Scikit-learn")
    st.sidebar.caption("SHAP · Groq · Plotly")
