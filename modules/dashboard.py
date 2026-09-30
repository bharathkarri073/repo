# modules/dashboard.py
"""
Executive dashboard components: hero banner, KPI cards, empty state,
footer. Public API is unchanged — render_hero(), render_kpi_card(),
calculate_dataset_health(), extract_best_metric(), render_dashboard_kpis(),
render_empty_state(), render_footer() all keep their existing signatures,
so streamlit_app.py needs no changes to pick this up.

render_dashboard_kpis() now also renders a secondary row of denser
".kpi-mini" cards (memory usage, duplicate %, training time, prediction
confidence, model complexity, data quality) automatically — no extra
call needed from the app.
"""

from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from textwrap import dedent


def render_hero():
    st.markdown("### ✦ Autonomous Analytics Workspace")
    st.title("GenAI Data Analyst System")
    st.caption(
        "Transform raw datasets into predictive intelligence using automated "
        "preprocessing, feature engineering, AutoML, explainable AI and "
        "real-time LLM business insights."
    )

    cols = st.columns(4)
    badges = [
        "⚡ Intelligent Processing",
        "🧠 AutoML",
        "🔎 Explainable AI",
        "🤖 Groq LLM",
    ]
    for col, badge in zip(cols, badges):
        col.markdown(f"<span style='display:inline-block; padding:6px 10px; border-radius:999px; background:rgba(148,163,184,0.12); color:#dfe7f3; font-size:0.78rem; margin:4px 0;'>{badge}</span>", unsafe_allow_html=True)


def _safe_metric(value, fallback="—"):
    if value is None:
        return fallback

    if isinstance(value, float) and np.isnan(value):
        return fallback

    return str(value)


def render_kpi_card(
    icon,
    value,
    label,
    status="LIVE",
):
    value = _safe_metric(value)
    label = str(label)
    status = str(status)

    st.markdown(
        f"""
        <div style="
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 14px 12px 12px;
            min-height: 150px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        ">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 12px;">
                <div style="font-size: 1.2rem;">{icon}</div>
                <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em; color: #7dd3fc; background: rgba(125,211,252,0.08); border-radius: 999px; padding: 4px 8px;">{status}</div>
            </div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #f8fafc; margin-bottom: 6px;">{value}</div>
            <div style="font-size: 0.82rem; color: #94a3b8;">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_mini(icon, value, label):
    """
    Denser secondary KPI card for supporting metrics that don't need
    the full-size treatment (memory usage, training time, etc).
    """

    value = _safe_metric(value)
    label = str(label)

    st.markdown(
        f"""
        <div style="
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid rgba(148, 163, 184, 0.15);
            border-radius: 14px;
            padding: 10px 12px;
            min-height: 72px;
        ">
            <div style="display:flex; align-items:center; gap:8px;">
                <span style="font-size:16px;">{icon}</span>
                <div>
                    <div style="font-size:1.1rem; font-weight:800; color:#f8fafc;">{value}</div>
                    <div style="font-size:0.72rem; color:#94a3b8;">{label}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_dataset_health(df):
    """
    Returns a simple data-quality score from 0 to 100.
    """

    if df is None or df.empty:
        return 0

    total_cells = max(df.shape[0] * df.shape[1], 1)

    missing_ratio = df.isna().sum().sum() / total_cells
    duplicate_ratio = df.duplicated().sum() / max(len(df), 1)

    constant_columns = sum(
        df[col].nunique(dropna=False) <= 1
        for col in df.columns
    )

    constant_ratio = constant_columns / max(len(df.columns), 1)

    penalty = (
        missing_ratio * 55
        + duplicate_ratio * 25
        + constant_ratio * 20
    )

    score = int(round(max(0, min(100, 100 - penalty * 100))))

    return score


def _format_memory(df):
    if df is None or df.empty:
        return "—"

    total_bytes = df.memory_usage(deep=True).sum()

    for unit in ["B", "KB", "MB", "GB"]:
        if total_bytes < 1024:
            return f"{total_bytes:.1f} {unit}"
        total_bytes /= 1024

    return f"{total_bytes:.1f} TB"


def _format_duplicate_pct(df):
    if df is None or df.empty:
        return "—"

    dup_ratio = df.duplicated().sum() / max(len(df), 1)

    return f"{dup_ratio * 100:.1f}%"


def _extract_prediction_confidence(results):
    """
    Reuses the same metric the AutoML leaderboard is sorted on
    (Accuracy for classification, R² for regression) as a rough
    confidence indicator, capped at 100%.
    """

    if not results:
        return None

    results_df = results.get("results")

    if results_df is None or results_df.empty:
        return None

    best_row = results_df.iloc[0]

    if results.get("problem_type") == "classification":
        value = best_row.get("Accuracy", np.nan)
    else:
        value = best_row.get("R² Score", np.nan)

    if pd.isna(value):
        return None

    return min(abs(float(value)) * 100, 100)


def extract_best_metric(results):
    """
    Extracts the main metric from the best row of AutoML results.
    """

    if not results:
        return "Not trained", "Model score"

    results_df = results.get("results")

    if results_df is None or results_df.empty:
        return "Unavailable", "Model score"

    best_row = results_df.iloc[0]

    if results.get("problem_type") == "classification":
        value = best_row.get("Accuracy", np.nan)

        if pd.notna(value):
            return f"{float(value) * 100:.2f}%", "Accuracy"

    else:
        value = best_row.get("R² Score", np.nan)

        if pd.notna(value):
            return f"{float(value):.4f}", "R² score"

    return "Unavailable", "Model score"


def render_dashboard_kpis(
    df,
    target_col=None,
    problem_type=None,
):
    results = st.session_state.get("automl_results")

    model_name = (
        results.get("best_model_name", "Not trained")
        if results
        else "Not trained"
    )

    metric_value, metric_label = extract_best_metric(results)

    health_score = calculate_dataset_health(df)

    row_count = len(df) if df is not None else 0
    feature_count = (
        max(df.shape[1] - 1, 0)
        if df is not None
        else 0
    )

    first_row = st.columns(5)

    with first_row[0]:
        render_kpi_card(
            "🗂️",
            f"{row_count:,}",
            "Dataset rows",
        )

    with first_row[1]:
        render_kpi_card(
            "🧩",
            feature_count,
            "Predictor features",
        )

    with first_row[2]:
        render_kpi_card(
            "💚",
            f"{health_score}%",
            "Dataset health",
        )

    with first_row[3]:
        render_kpi_card(
            "🏆",
            model_name,
            "Best model",
            status="TRAINED" if results else "PENDING",
        )

    with first_row[4]:
        render_kpi_card(
            "🎯",
            metric_value,
            metric_label,
            status=(
                problem_type.upper()
                if problem_type
                else "AUTO"
            ),
        )

    if target_col:
        st.caption(
            f"Active target: {target_col} · Problem type: {(problem_type or 'Unknown').title()}"
        )

    _render_secondary_kpis(df, results)


def _render_secondary_kpis(df, results):
    """
    Denser row of supporting metrics beneath the primary KPI cards.
    Falls back to '—' for anything not yet available (e.g. before
    AutoML has run) rather than raising or hiding the row entirely.
    """

    memory_usage = _format_memory(df)
    duplicate_pct = _format_duplicate_pct(df)

    training_seconds = st.session_state.get("automl_training_seconds")
    training_time = (
        f"{training_seconds:.1f}s" if training_seconds is not None else "—"
    )

    confidence = _extract_prediction_confidence(results)
    confidence_display = f"{confidence:.1f}%" if confidence is not None else "—"

    complexity = results.get("feature_names") if results else None
    model_complexity = f"{len(complexity)} features" if complexity is not None else "—"

    second_row = st.columns(5)

    with second_row[0]:
        render_kpi_mini("💾", memory_usage, "Memory usage")

    with second_row[1]:
        render_kpi_mini("👥", duplicate_pct, "Duplicate rows")

    with second_row[2]:
        render_kpi_mini("⏱️", training_time, "Training time")

    with second_row[3]:
        render_kpi_mini("📶", confidence_display, "Prediction confidence")

    with second_row[4]:
        render_kpi_mini("🧠", model_complexity, "Model complexity")

    st.markdown("&nbsp;", unsafe_allow_html=True)


def render_empty_state():
    col_icon, col_text = st.columns([1, 4])
    with col_icon:
        st.markdown("📂")
    with col_text:
        st.subheader("Connect your dataset")
        st.caption(
            "Upload a CSV file from the AI Control Center to activate preprocessing, "
            "analytics, AutoML, explainability and GenAI insights."
        )


def render_footer():
    st.markdown("---")
    st.caption("• Powered by Streamlit · Scikit-learn · SHAP · Groq AI")
