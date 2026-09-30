"""
PART 4 — Intelligent Numeric Feature Engineering
PART 5 — Smart Interaction Feature Generation

Combined into one file since Part 5 operates on the numeric output
of Part 4 and both are relatively short.

BUG FIX vs the original modules/feature_engineering.py: in the
skewness-transform loop below, the
    report.append({..., "Technique": "Square Root Transformation", ...})
call was indented at the same level as the `if`/`elif` that decides
whether a transform even happened — so it logged "Square Root
Transformation" for every numeric column on every run, even columns
that got log-transformed instead, or weren't transformed at all.
It's now nested inside the `elif` so it only logs when that branch
actually executes.
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import f_regression, f_classif


# ============================================================
# PART 4 — Intelligent Numeric Feature Engineering
# ============================================================

def intelligent_numeric_engineering(df, target_col=None):

    df = df.copy()
    report = []

    # --------------------------------------------------------
    # Preserve Target
    # --------------------------------------------------------

    if target_col is not None and target_col in df.columns:
        target = df[target_col]
        feature_df = df.drop(columns=[target_col])
    else:
        feature_df = df.copy()
        target = None

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    for col in numeric_cols:

        try:

            # --------------------------------------------
            # Replace Infinite Values
            # --------------------------------------------

            feature_df[col] = feature_df[col].replace([np.inf, -np.inf], np.nan)
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

            # --------------------------------------------
            # Skewness
            # --------------------------------------------

            skewness = feature_df[col].skew()

            if abs(skewness) > 1:

                # Highly skewed -> log transform
                shift = 0
                if feature_df[col].min() <= 0:
                    shift = abs(feature_df[col].min()) + 1

                feature_df[col] = np.log1p(feature_df[col] + shift)
                new_skew = feature_df[col].skew()

                report.append({
                    "Feature": col,
                    "Technique": "Log Transformation",
                    "Details": f"{round(skewness, 2)} -> {round(new_skew, 2)}",
                })

            elif abs(skewness) > 0.5:

                # Moderately skewed -> square root transform
                shift = 0
                if feature_df[col].min() < 0:
                    shift = abs(feature_df[col].min())

                feature_df[col] = np.sqrt(feature_df[col] + shift)

                # Fixed: this report line now only fires for columns
                # that actually went through the sqrt transform above,
                # not unconditionally for every numeric column.
                report.append({
                    "Feature": col,
                    "Technique": "Square Root Transformation",
                    "Details": f"Skewness={round(skewness, 2)}",
                })

            # --------------------------------------------
            # Winsorization
            # --------------------------------------------

            q1 = feature_df[col].quantile(0.25)
            q3 = feature_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            original = feature_df[col].copy()
            feature_df[col] = feature_df[col].clip(lower, upper)

            changed = (original != feature_df[col]).sum()

            if changed > 0:
                report.append({
                    "Feature": col,
                    "Technique": "Outlier Clipping",
                    "Details": f"{changed} values clipped",
                })

        except Exception:
            pass

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    if target is not None:
        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df, report_df)


# ============================================================
# PART 5 — Smart Interaction Feature Generation
# ============================================================

def smart_interaction_features(df, target_col, problem_type):

    df = df.copy()
    report = []

    # --------------------------------------------------------
    # Preserve Target
    # --------------------------------------------------------

    target = df[target_col]
    feature_df = df.drop(columns=[target_col])

    # --------------------------------------------------------
    # Numeric Features
    # --------------------------------------------------------

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        feature_df[target_col] = target
        return feature_df, pd.DataFrame(report)

    # --------------------------------------------------------
    # Select Most Important Features
    # --------------------------------------------------------

    try:

        X = feature_df[numeric_cols]
        y = target

        if problem_type == "regression":
            scores, _ = f_regression(X, y)
        else:
            scores, _ = f_classif(X, y)

        importance = pd.DataFrame({"Feature": numeric_cols, "Score": scores})
        importance = importance.sort_values(by="Score", ascending=False)
        importance.reset_index(drop=True, inplace=True)

        top_features = importance["Feature"].head(5).tolist()

    except Exception:

        top_features = numeric_cols[:min(5, len(numeric_cols))]

    # --------------------------------------------------------
    # Multiplication Features
    # --------------------------------------------------------

    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]
            col2 = top_features[j]
            new_col = f"{col1}_x_{col2}"

            feature_df[new_col] = feature_df[col1] * feature_df[col2]

            report.append({
                "Feature": new_col,
                "Technique": "Interaction Feature",
                "Details": f"{col1} × {col2}",
            })

    # --------------------------------------------------------
    # Ratio Features
    # --------------------------------------------------------

    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]
            col2 = top_features[j]
            new_col = f"{col1}_div_{col2}"

            denominator = feature_df[col2].replace(0, np.nan)
            feature_df[new_col] = feature_df[col1] / denominator
            feature_df[new_col] = feature_df[new_col].fillna(0)

            report.append({
                "Feature": new_col,
                "Technique": "Ratio Feature",
                "Details": f"{col1} / {col2}",
            })

    # --------------------------------------------------------
    # Difference Features
    # --------------------------------------------------------

    for i in range(len(top_features)):
        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]
            col2 = top_features[j]
            new_col = f"{col1}_minus_{col2}"

            feature_df[new_col] = feature_df[col1] - feature_df[col2]

            report.append({
                "Feature": new_col,
                "Technique": "Difference Feature",
                "Details": f"{col1} - {col2}",
            })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target
    report_df = pd.DataFrame(report)

    return (feature_df, report_df)
