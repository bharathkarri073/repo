"""
PART 1 — Intelligent Feature Selection

Removes duplicate columns, constant columns, near-constant columns,
low-variance numeric columns, and identifier columns. Logic is
unchanged from the original modules/feature_engineering.py — only
relocated into its own file.
"""

import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold


def intelligent_feature_selection(df, target_col=None):

    df = df.copy()
    report = []

    # --------------------------------------------------------
    # Preserve target
    # --------------------------------------------------------

    if target_col is not None and target_col in df.columns:
        target = df[target_col]
        feature_df = df.drop(columns=[target_col])
    else:
        feature_df = df.copy()
        target = None

    # ========================================================
    # 1. Remove Duplicate Columns
    # ========================================================

    duplicate_columns = []
    columns = feature_df.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            c1 = columns[i]
            c2 = columns[j]
            try:
                if feature_df[c1].equals(feature_df[c2]):
                    duplicate_columns.append(c2)
            except Exception:
                pass

    if duplicate_columns:
        feature_df.drop(columns=duplicate_columns, inplace=True)
        for col in duplicate_columns:
            report.append({
                "Feature": col,
                "Technique": "Duplicate Feature Removal",
                "Details": "Duplicate of another feature",
            })

    # ========================================================
    # 2. Remove Constant Features
    # ========================================================

    constant_features = [
        col for col in feature_df.columns
        if feature_df[col].nunique(dropna=False) <= 1
    ]

    if constant_features:
        feature_df.drop(columns=constant_features, inplace=True)
        for col in constant_features:
            report.append({
                "Feature": col,
                "Technique": "Constant Feature Removal",
                "Details": "Only one unique value",
            })

    # ========================================================
    # 3. Remove Near Constant Features
    # ========================================================

    near_constant = []
    threshold = 0.98

    for col in feature_df.columns:
        try:
            top_frequency = feature_df[col].value_counts(normalize=True).iloc[0]
            if top_frequency > threshold:
                near_constant.append(col)
        except Exception:
            pass

    if near_constant:
        feature_df.drop(columns=near_constant, inplace=True)
        for col in near_constant:
            report.append({
                "Feature": col,
                "Technique": "Near Constant Removal",
                "Details": "Dominated by one value",
            })

    # ========================================================
    # 4. Remove Low Variance Numeric Features
    # ========================================================

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        selector = VarianceThreshold(threshold=0.01)
        selector.fit(feature_df[numeric_cols])

        selected = [
            c for c, keep in zip(numeric_cols, selector.get_support()) if keep
        ]

        removed = [col for col in numeric_cols if col not in selected]

        if removed:
            feature_df.drop(columns=removed, inplace=True)
            for col in removed:
                report.append({
                    "Feature": col,
                    "Technique": "Low Variance Removal",
                    "Details": "Variance < 0.01",
                })

    # ========================================================
    # 5. Remove ID Columns
    # ========================================================

    id_keywords = [
        "id", "index", "customerid", "userid",
        "employeeid", "transactionid", "recordid",
    ]

    id_columns = []

    for col in feature_df.columns:

        lower = col.lower().replace("_", "")

        if lower in id_keywords:
            id_columns.append(col)
            continue

        if (
            feature_df[col].nunique() == len(feature_df)
            and "id" in col.lower().replace("_", "")
        ):
            if feature_df[col].dtype != "float64":
                id_columns.append(col)

    if id_columns:
        feature_df.drop(columns=id_columns, inplace=True)
        for col in id_columns:
            report.append({
                "Feature": col,
                "Technique": "Identifier Removal",
                "Details": "Unique identifier",
            })

    # ========================================================
    # 6. Target Leakage & Collinear Proxy Removal
    # ========================================================

    if target is not None:
        leakage_columns = []
        target_clean = str(target_col).lower().replace("_", " ").strip()

        # Check for financial / stock price twin columns (e.g. Close vs Adj Close)
        stock_twin_groups = [
            {"close", "adj close", "adjusted close", "close price", "adjclose"},
            {"price", "last price", "current price"},
        ]

        for col in list(feature_df.columns):
            col_clean = str(col).lower().replace("_", " ").strip()

            # Check known twin pairs
            is_twin = False
            for group in stock_twin_groups:
                if target_clean in group and col_clean in group and target_clean != col_clean:
                    is_twin = True
                    break

            if is_twin:
                leakage_columns.append((col, "Target Twin (e.g. Close / Adj Close)"))
                continue

            # Check extreme numerical correlation with target
            if pd.api.types.is_numeric_dtype(feature_df[col]) and pd.api.types.is_numeric_dtype(target):
                try:
                    s_feat = feature_df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                    s_targ = target.replace([np.inf, -np.inf], np.nan).fillna(0)
                    if s_feat.std() > 0 and s_targ.std() > 0:
                        corr = np.corrcoef(s_feat, s_targ)[0, 1]
                        if abs(corr) >= 0.95:
                            leakage_columns.append((
                                col,
                                f"Extreme Target Correlation (|r|={abs(corr):.4f} >= 0.95)"
                            ))
                except Exception:
                    pass

        if leakage_columns:
            drop_cols = [c[0] for c in leakage_columns]
            feature_df.drop(columns=drop_cols, inplace=True)
            for col, reason in leakage_columns:
                report.append({
                    "Feature": col,
                    "Technique": "Target Leakage Guardrail",
                    "Details": reason,
                })

    # ========================================================
    # Restore Target
    # ========================================================

    if target is not None:
        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df, report_df)
