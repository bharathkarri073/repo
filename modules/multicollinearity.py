"""
PART 8 — Intelligent Multicollinearity Removal (Correlation + VIF)

First drops one side of any highly correlated pair (keeping whichever
scored higher in ranking_df's Combined Score), then iteratively drops
the single highest-VIF feature until everything is under
vif_threshold. Logic unchanged from the original file.
"""

import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor


def intelligent_multicollinearity_removal(
    df, target_col, ranking_df, corr_threshold=0.90, vif_threshold=10
):

    df = df.copy()
    report = []

    # --------------------------------------------------------
    # Preserve Target
    # --------------------------------------------------------

    target = df[target_col]
    feature_df = df.drop(columns=[target_col])

    # --------------------------------------------------------
    # Numeric Features Only
    # --------------------------------------------------------

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) <= 1:
        feature_df[target_col] = target
        return (feature_df, pd.DataFrame(), pd.DataFrame())

    # --------------------------------------------------------
    # Feature Importance Dictionary
    # --------------------------------------------------------

    importance = dict(zip(ranking_df["Feature"], ranking_df["Combined Score"]))

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    corr_matrix = (
        feature_df[numeric_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .corr()
        .abs()
    )

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    remove_columns = set()

    # --------------------------------------------------------
    # Remove Less Important Correlated Feature
    # --------------------------------------------------------

    for col in upper.columns:

        correlated = upper.index[upper[col] > corr_threshold].tolist()

        for other in correlated:

            score_col = importance.get(col, 0)
            score_other = importance.get(other, 0)

            if score_col >= score_other:
                remove_columns.add(other)
            else:
                remove_columns.add(col)

    if remove_columns:

        feature_df.drop(columns=list(remove_columns), inplace=True, errors="ignore")

        for col in remove_columns:
            report.append({
                "Feature": col,
                "Technique": "Correlation Removal",
                "Details": f"Correlation > {corr_threshold}",
            })

    # --------------------------------------------------------
    # VIF Calculation (iterative)
    # --------------------------------------------------------

    vif_df = pd.DataFrame()

    while True:

        numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) <= 1:
            break

        vif_df = pd.DataFrame()
        vif_df["Feature"] = numeric_cols

        vif_df["VIF"] = [
            variance_inflation_factor(
                feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values,
                i,
            )
            for i in range(len(numeric_cols))
        ]

        vif_df = vif_df.sort_values(by="VIF", ascending=False).reset_index(drop=True)

        max_vif = vif_df["VIF"].max()

        if max_vif <= vif_threshold:
            break

        remove_feature = vif_df.iloc[0]["Feature"]
        feature_df.drop(columns=[remove_feature], inplace=True)

        report.append({
            "Feature": remove_feature,
            "Technique": "VIF Removal",
            "Details": f"VIF={round(max_vif, 2)}",
        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target
    report_df = pd.DataFrame(report)

    return (feature_df, vif_df, report_df)
