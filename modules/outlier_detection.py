import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

PLOTLY_DARK_LAYOUT = dict(
    plot_bgcolor="#0a0f1a",
    paper_bgcolor="#0a0f1a",
    font=dict(family="Inter, sans-serif", color="#c0d4f0"),
)

SKIP_KEYWORDS = ['id', 'zip', 'zipcode', 'postal', 'latitude', 'longitude']


# ==========================================================
# Pure logic (no Streamlit calls) — safe to unit test / reuse
# ==========================================================

def _select_treatable_columns(df, numeric_cols):
    """
    Splits numeric columns into ones eligible for outlier treatment
    and ones skipped because they're identifiers/geospatial (treating
    a zip code or lat/long as an outlier-prone measurement produces
    nonsense results).
    """

    valid_cols, skipped_cols = [], []

    for col in numeric_cols:
        if any(word in col.lower() for word in SKIP_KEYWORDS):
            skipped_cols.append(col)
        else:
            valid_cols.append(col)

    return valid_cols, skipped_cols


def detect_and_treat_column(df, col):
    """
    Pure logic for a single column: decides a treatment strategy and
    applies it, returning (updated_df, summary_dict). Separated from
    rendering so the same decision logic can be reused by future
    programmatic callers (e.g. a batch/report mode) without dragging
    Streamlit widgets along.
    """

    df = df.copy()
    data = df[col].dropna()

    if len(data) == 0:
        return df, None

    skewness = data.skew()
    unique_ratio = data.nunique() / len(data)
    is_count_variable = (
        pd.api.types.is_integer_dtype(data) and unique_ratio < 0.05
    )

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = (data < lower) | (data > upper)
    outlier_count = int(outliers.sum())
    outlier_pct = round((outlier_count / len(data)) * 100, 2)

    if outlier_count == 0:
        treatment = "No Treatment Needed"
        reason = "No significant outliers detected."

    elif is_count_variable:
        df[col] = pd.cut(df[col], bins=5, labels=False)
        treatment = "Binning"
        reason = "Count variable detected. Binning used to reduce skewness and extreme values."

    elif abs(skewness) > 1:
        shift_value = abs(df[col].min()) + 1
        df[col] = np.log1p(df[col] + shift_value)
        treatment = "Log Transformation"
        reason = "Highly skewed distribution detected. Log transformation applied."

    else:
        df = df[~((df[col] < lower) | (df[col] > upper))]
        treatment = "Outliers Removed"
        reason = "Moderate outliers detected. IQR-based removal applied."

    problem_detected = (
        "High Skewness + Extreme Outliers" if abs(skewness) > 1 else "Moderate Outliers"
    )

    summary = {
        "Column": col,
        "Problem Detected": problem_detected,
        "Outlier Count": outlier_count,
        "Outlier %": outlier_pct,
        "Treatment": treatment,
        "Reason": reason,
    }

    return df, summary


# ==========================================================
# Rendering
# ==========================================================

def _render_box_comparison(before_df, after_df, col, idx):

    left, right = st.columns(2)

    with left:
        fig_before = px.box(
            before_df, y=col,
            title=f"{col} — Before", color_discrete_sequence=["#ff6b6b"],
        )
        fig_before.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
        st.plotly_chart(fig_before, use_container_width=True, key=f"outlier_before_{col}_{idx}")

    with right:
        fig_after = px.box(
            after_df, y=col,
            title=f"{col} — After", color_discrete_sequence=["#7cb9e8"],
        )
        fig_after.update_layout(**PLOTLY_DARK_LAYOUT, height=340)
        st.plotly_chart(fig_after, use_container_width=True, key=f"outlier_after_{col}_{idx}")


def intelligent_outlier_treatment(df):
    """
    UI-facing entry point. Signature and return value are unchanged:
    (df_processed, outlier_df). Internally now delegates the per-column
    decision to detect_and_treat_column() so the logic isn't tangled
    with the plotting/layout code.
    """

    st.markdown(
        '<div class="dashboard-card">'
        '<div class="section-title">🚨 Intelligent Outlier Detection & Treatment</div>'
        '<div class="section-caption">IQR-based detection with automatic strategy selection '
        '(binning, log transform, or removal) per column.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    valid_cols, skipped_cols = _select_treatable_columns(df_processed, numeric_cols)

    st.markdown(
        '<div class="section-title" style="margin-top:14px;">🎯 Select Columns for Outlier Treatment</div>',
        unsafe_allow_html=True,
    )

    selected_cols = st.multiselect(
        "Choose columns", valid_cols, default=valid_cols,
        key="outlier_treatment_multiselect",
    )

    outlier_results = []

    for idx, col in enumerate(selected_cols):

        before_snapshot = df_processed[[col]].copy()

        df_processed, summary = detect_and_treat_column(df_processed, col)

        if summary is None:
            continue

        with st.expander(f"📦 {col} — {summary['Treatment']}", expanded=False):
            _render_box_comparison(before_snapshot, df_processed[[col]] if col in df_processed.columns else before_snapshot, col, idx)
            st.caption(summary["Reason"])

        outlier_results.append(summary)

    outlier_df = pd.DataFrame(outlier_results)

    st.markdown(
        '<div class="section-title" style="margin-top:10px;">📋 Outlier Detection Summary</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(outlier_df, use_container_width=True)

    if skipped_cols:
        st.info(f"Skipped identifier/geospatial columns: {', '.join(skipped_cols)}")

    return df_processed, outlier_df
