"""
PART 9 — Intelligent Scaling + PCA

Applies StandardScaler/MinMaxScaler to numeric columns, and
optionally reduces them to principal components retaining a target
explained-variance ratio. Logic unchanged from the original file.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def intelligent_scaling_pca(
    df, target_col, scaling_method="StandardScaler", apply_pca=False, variance_threshold=0.95
):

    df = df.copy()
    report = []

    # --------------------------------------------------------
    # Preserve Target
    # --------------------------------------------------------

    target = df[target_col].copy()
    feature_df = df.drop(columns=[target_col]).copy()

    # --------------------------------------------------------
    # Numeric Columns
    # --------------------------------------------------------

    numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) == 0:
        feature_df[target_col] = target
        return (feature_df, pd.DataFrame(report))

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:

        feature_df[numeric_cols] = (
            feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        feature_df[numeric_cols] = scaler.fit_transform(feature_df[numeric_cols])

        report.append({
            "Feature": "Numeric Features",
            "Technique": scaling_method,
            "Details": "Scaling Applied",
        })

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    if apply_pca and len(numeric_cols) > 1:

        pca = PCA(n_components=variance_threshold, random_state=42)

        transformed = pca.fit_transform(
            feature_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        pca_columns = [f"PC{i + 1}" for i in range(transformed.shape[1])]

        pca_df = pd.DataFrame(transformed, columns=pca_columns, index=feature_df.index)

        # Drop original numeric features
        feature_df.drop(columns=numeric_cols, inplace=True)
        feature_df = pd.concat([feature_df, pca_df], axis=1)

        explained = round(np.sum(pca.explained_variance_ratio_) * 100, 2)

        report.append({
            "Feature": "Principal Components",
            "Technique": "PCA",
            "Details": f"{len(pca_columns)} PCs retained ({explained}% variance explained)",
        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target
    report_df = pd.DataFrame(report)

    return (feature_df, report_df)
