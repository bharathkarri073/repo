import streamlit as st
import pandas as pd
import numpy as np

from modules.outlier_detection import intelligent_outlier_treatment


DATETIME_KEYWORDS = [
    'date', 'time', 'timestamp',
    'created', 'updated', 'year',
    'month', 'day'
]

def _impute_datetime_column(df, col, missing_pct):
    """
    Returns (df, method_used, reason). Mirrors the original three-tier
    logic exactly (<=10% / <=40% / >40% missing) — only extracted into
    its own function so intelligent_preprocessing() doesn't have one
    giant branching body.
    """

    if missing_pct <= 10:
        df[col] = df[col].ffill().bfill()
        return df, "Time Interpolation + Forward/Backward Fill", (
            "Small datetime gaps detected. Applied interpolation and sequential fill."
        )

    if missing_pct <= 40:
        df[col] = df[col].ffill().bfill()
        return df, "Forward Fill + Backward Fill", (
            "Moderate datetime gaps detected. Sequential datetime filling applied."
        )

    if any(keyword in col.lower() for keyword in DATETIME_KEYWORDS):
        df[col] = df[col].ffill().bfill()
        return df, "Retained Important Datetime Column", (
            "Datetime column considered important. Retained and filled sequentially."
        )

    df.drop(columns=[col], inplace=True)
    return df, "Dropped Datetime Column", (
        "High missing percentage and column not considered critical."
    )


def _impute_column(df, col, numeric_cols, categorical_cols):
    """
    Returns (df, method_used, reason) for one column, applying the
    same rules as the original inline logic: drop if >60% missing,
    datetime-aware fill, mean/median for numeric depending on
    skewness, mode for categorical.
    """

    missing_pct = (df[col].isnull().sum() / len(df)) * 100

    if missing_pct == 0:
        return df, "", ""

    if missing_pct > 60 and not pd.api.types.is_datetime64_any_dtype(df[col]):
        df.drop(columns=[col], inplace=True)
        return df, "Drop Column", "Column has more than 60% missing values."

    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return _impute_datetime_column(df, col, missing_pct)

    if col in numeric_cols:
        skewness = df[col].skew()
        if abs(skewness) < 0.5:
            df[col] = df[col].fillna(df[col].mean())
            return df, "Mean Imputation", "Normally distributed numeric data."
        else:
            df[col] = df[col].fillna(df[col].median())
            return df, "Median Imputation", "Skewed numeric data with outliers."

    if col in categorical_cols:
        mode_series = df[col].mode()
        mode_value = mode_series[0] if not mode_series.empty else "Unknown"
        df[col] = df[col].fillna(mode_value)
        return df, "Mode Imputation", "Categorical/string column."

    return df, "", ""


def run_missing_value_imputation(df):
    """
    Pure logic pass over every column. Returns
    (df, recommendations_df) — recommendations_df is the transformation
    log: one row per column that actually needed imputation, recording
    what was done and why.
    """

    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    log_rows = []

    for col in list(df.columns):

        missing_pct = (df[col].isnull().sum() / len(df)) * 100

        df, method_used, reason = _impute_column(df, col, numeric_cols, categorical_cols)

        if missing_pct > 0:
            log_rows.append({
                "Column": col,
                "Missing %": round(missing_pct, 2),
                "Method Used": method_used,
                "Reason": reason,
            })

    return df, pd.DataFrame(log_rows)


def compute_skewness_report(df):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    rows = []

    for col in numeric_cols:
        skew_val = df[col].skew()

        if abs(skew_val) < 0.5:
            interpretation = "Approximately Symmetric"
        elif abs(skew_val) < 1:
            interpretation = "Moderately Skewed"
        else:
            interpretation = "Highly Skewed"

        rows.append({
            "Column": col,
            "Skewness": round(skew_val, 2),
            "Interpretation": interpretation,
        })

    return pd.DataFrame(rows)


def compute_imputation_confirmation(df):

    rows = [
        {"Column": col, "Missing Values After Imputation": int(df[col].isnull().sum())}
        for col in df.columns
        if df[col].isnull().sum() == 0
    ]

    return pd.DataFrame(rows)


# ==========================================================
# Rendering
# ==========================================================

def _render_before_after_card(before_shape, after_shape):

    rows_before, cols_before = before_shape
    rows_after, cols_after = after_shape

    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="section-title">📐 Before vs After</div>
            <div style="display:flex; gap:32px; margin-top:10px;">
                <div>
                    <div style="color:#94a3b8; font-size:0.78rem;">Rows</div>
                    <div style="color:#f8fafc; font-size:1.3rem; font-weight:800;">
                        {rows_before:,} → {rows_after:,}
                    </div>
                </div>
                <div>
                    <div style="color:#94a3b8; font-size:0.78rem;">Columns</div>
                    <div style="color:#f8fafc; font-size:1.3rem; font-weight:800;">
                        {cols_before} → {cols_after}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def intelligent_preprocessing(df):
    """
    UI-facing entry point. Signature and return value are unchanged:
    (df_processed, recommendations_df, imputation_df, skewness_df,
    outlier_df). Internally now delegates to the pure-logic functions
    above so the decision rules are testable independent of Streamlit.
    """

    st.markdown(
        '<div class="dashboard-card">'
        '<div class="section-title">📑 Automated Data Refinement System</div>'
        '<div class="section-caption">Missing-value imputation, skewness analysis and outlier '
        'treatment, chosen automatically per column.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    original_shape = df.shape

    df_processed, recommendations_df = run_missing_value_imputation(df)

    imputation_df = compute_imputation_confirmation(df_processed)
    skewness_df = compute_skewness_report(df_processed)

    df_processed, outlier_df = intelligent_outlier_treatment(df_processed)

    _render_before_after_card(original_shape, df_processed.shape)

    return (df_processed, recommendations_df, imputation_df, skewness_df, outlier_df)
