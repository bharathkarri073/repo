"""
PART 7 — Mutual Information + Random Forest Feature Importance +
Automatic Feature Selection

Combines normalized mutual information and Random Forest importance
into one "Combined Score" per feature, keeps the top N, and reports
which features were dropped. Logic unchanged from the original file.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def intelligent_feature_ranking(df, target_col, problem_type, top_features=30):

    df = df.copy()
    report = []

    target = df[target_col]
    feature_df = df.drop(columns=[target_col])

    # --------------------------------------------------------
    # Keep only numeric columns
    # --------------------------------------------------------

    numeric_df = feature_df.select_dtypes(include=np.number)

    if numeric_df.empty:
        df[target_col] = target
        return (df, pd.DataFrame(), pd.DataFrame())

    X = numeric_df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = target

    # --------------------------------------------------------
    # Mutual Information + Random Forest Importance
    # --------------------------------------------------------

    if problem_type == "classification":
        mi_scores = mutual_info_classif(X, y, random_state=42)
        model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    else:
        mi_scores = mutual_info_regression(X, y, random_state=42)
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)

    model.fit(X, y)
    rf_scores = model.feature_importances_

    # --------------------------------------------------------
    # Normalize Scores
    # --------------------------------------------------------

    scaler = MinMaxScaler()
    mi_scores = scaler.fit_transform(mi_scores.reshape(-1, 1)).flatten()
    rf_scores = scaler.fit_transform(rf_scores.reshape(-1, 1)).flatten()

    # --------------------------------------------------------
    # Combined Score
    # --------------------------------------------------------

    ranking_df = pd.DataFrame({
        "Feature": X.columns,
        "Mutual Information": mi_scores,
        "Random Forest Importance": rf_scores,
    })

    ranking_df["Combined Score"] = (
        ranking_df["Mutual Information"] + ranking_df["Random Forest Importance"]
    ) / 2

    ranking_df = ranking_df.sort_values(by="Combined Score", ascending=False)
    ranking_df.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # Select Top Features
    # --------------------------------------------------------

    selected_features = ranking_df.head(
        min(top_features, len(ranking_df))
    )["Feature"].tolist()

    final_df = feature_df[selected_features].copy()
    final_df[target_col] = target

    # --------------------------------------------------------
    # Report Removed Features
    # --------------------------------------------------------

    removed = sorted([col for col in feature_df.columns if col not in selected_features])

    for col in removed:
        report.append({
            "Feature": col,
            "Technique": "Feature Selection",
            "Details": "Removed due to low combined importance",
        })

    report_df = pd.DataFrame(report)

    return (final_df, ranking_df, report_df)
