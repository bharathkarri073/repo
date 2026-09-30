"""
PART 10 — Master Feature Engineering Pipeline

Orchestrates parts 1-9 in sequence and concatenates their reports
into one final DataFrame. Public signature is unchanged:

    intelligent_feature_engineering(df, target_col, problem_type,
        scaling_method="StandardScaler", apply_pca=False,
        variance_threshold=0.95)
    -> (df, ranking_df, vif_df, final_report)

streamlit_app.py calls this with keyword arguments, so the change
from a single 2,144-line file to this package is fully transparent
to callers.
"""

import pandas as pd

from .selection import intelligent_feature_selection
from .datetime_features import datetime_feature_engineering
from .categorical import intelligent_categorical_engineering
from .numeric import intelligent_numeric_engineering, smart_interaction_features
from .polynomial import intelligent_polynomial_features
from .ranking import intelligent_feature_ranking
from .multicollinearity import intelligent_multicollinearity_removal
from .scaling_pca import intelligent_scaling_pca


def intelligent_feature_engineering(
    df,
    target_col,
    problem_type,
    scaling_method="StandardScaler",
    apply_pca=False,
    variance_threshold=0.95,
):

    df = df.copy()
    engineering_reports = []

    ranking_df = pd.DataFrame()
    vif_df = pd.DataFrame()

    # PART 1 — Feature selection (duplicates, constants, IDs, low variance)
    df, report = intelligent_feature_selection(df, target_col)
    engineering_reports.append(report)

    # PART 2 — Datetime feature engineering
    df, report = datetime_feature_engineering(df, target_col)
    engineering_reports.append(report)

    # PART 3 — Categorical encoding
    df, report = intelligent_categorical_engineering(df, target_col, problem_type)
    engineering_reports.append(report)

    # PART 4 — Numeric transforms (skew correction, winsorization)
    df, report = intelligent_numeric_engineering(df, target_col)
    engineering_reports.append(report)

    # PART 5 — Interaction features
    df, report = smart_interaction_features(df, target_col, problem_type)
    engineering_reports.append(report)

    # PART 6 — Polynomial features
    df, report = intelligent_polynomial_features(df, target_col, problem_type)
    engineering_reports.append(report)

    # PART 7 — Feature ranking + selection
    df, ranking_df, report = intelligent_feature_ranking(df, target_col, problem_type)
    engineering_reports.append(report)

    # PART 8 — Multicollinearity removal (correlation + VIF)
    df, vif_df, report = intelligent_multicollinearity_removal(df, target_col, ranking_df)
    engineering_reports.append(report)

    # PART 9 — Scaling + PCA
    df, report = intelligent_scaling_pca(
        df, target_col, scaling_method, apply_pca, variance_threshold
    )
    engineering_reports.append(report)

    # --------------------------------------------------------
    # Final Report
    # --------------------------------------------------------

    engineering_reports = [
        report for report in engineering_reports
        if report is not None and not report.empty
    ]

    if engineering_reports:
        final_report = pd.concat(engineering_reports, ignore_index=True)
    else:
        final_report = pd.DataFrame()

    return (df, ranking_df, vif_df, final_report)
