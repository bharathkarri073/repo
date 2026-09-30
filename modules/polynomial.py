"""
PART 6 — Intelligent Polynomial Feature Generation

Selects the top-scoring numeric features (via f_regression/f_classif)
and generates polynomial combinations of them, keeping only the newly
generated columns. Logic unchanged from the original file.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import f_regression, f_classif


def intelligent_polynomial_features(df, target_col, problem_type, max_features=5, degree=2):

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

    if len(numeric_cols) == 0:
        feature_df[target_col] = target
        return feature_df, pd.DataFrame(report)

    # --------------------------------------------------------
    # Safe features (exclude any target proxies before polynomial explosion)
    # --------------------------------------------------------

    safe_numeric_cols = []
    for c in numeric_cols:
        try:
            if pd.api.types.is_numeric_dtype(target):
                s_c = feature_df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
                s_t = target.replace([np.inf, -np.inf], np.nan).fillna(0)
                if s_c.std() > 0 and s_t.std() > 0:
                    r = np.corrcoef(s_c, s_t)[0, 1]
                    if abs(r) >= 0.90:
                        continue
        except Exception:
            pass
        safe_numeric_cols.append(c)

    if len(safe_numeric_cols) == 0:
        feature_df[target_col] = target
        return feature_df, pd.DataFrame(report)

    # --------------------------------------------------------
    # Select Top Important Features
    # --------------------------------------------------------

    try:

        X = feature_df[safe_numeric_cols]
        y = target

        if problem_type == "regression":
            scores, _ = f_regression(X, y)
        else:
            scores, _ = f_classif(X, y)

        importance = pd.DataFrame({"Feature": safe_numeric_cols, "Score": scores})
        importance = importance.sort_values(by="Score", ascending=False)
        importance.reset_index(drop=True, inplace=True)

        selected_features = importance["Feature"].head(max_features).tolist()

    except Exception:

        selected_features = safe_numeric_cols[:min(max_features, len(safe_numeric_cols))]

    # --------------------------------------------------------
    # Polynomial Features
    # --------------------------------------------------------

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    transformed = poly.fit_transform(feature_df[selected_features])
    poly_names = poly.get_feature_names_out(selected_features)

    poly_df = pd.DataFrame(transformed, columns=poly_names, index=feature_df.index)

    # --------------------------------------------------------
    # Keep ONLY newly generated columns
    # --------------------------------------------------------

    new_columns = [col for col in poly_df.columns if col not in selected_features]

    if len(new_columns) == 0:
        feature_df[target_col] = target
        return feature_df, pd.DataFrame(report)

    poly_df = poly_df[new_columns]
    feature_df = pd.concat([feature_df, poly_df], axis=1)

    # --------------------------------------------------------
    # Engineering Report
    # --------------------------------------------------------

    for col in new_columns:
        report.append({
            "Feature": col,
            "Technique": "Polynomial Feature",
            "Details": f"Degree {degree}",
        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target
    report_df = pd.DataFrame(report)

    return (feature_df, report_df)
