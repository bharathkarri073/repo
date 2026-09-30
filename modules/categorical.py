"""
PART 3 — Intelligent Categorical Engineering

Encodes categorical columns based on cardinality: one-hot for low
cardinality (<=10), label encoding for medium (<=30), frequency
encoding for high. Adds mean target encoding on top for
classification problems. Logic unchanged from the original file.
"""

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


def intelligent_categorical_engineering(df, target_col=None, problem_type=None):

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

    # --------------------------------------------------------
    # Find categorical columns
    # --------------------------------------------------------

    categorical_cols = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Encode each categorical feature
    # --------------------------------------------------------

    for col in categorical_cols:

        unique_count = feature_df[col].nunique()

        # LOW CARDINALITY -> One-Hot Encoding
        if unique_count <= 10:

            onehot = pd.get_dummies(feature_df[col], prefix=col, drop_first=True)
            feature_df.drop(columns=[col], inplace=True)
            feature_df = pd.concat([feature_df, onehot], axis=1)

            report.append({
                "Feature": col,
                "Technique": "One-Hot Encoding",
                "Details": f"{unique_count} categories",
            })

        # MEDIUM CARDINALITY -> Label Encoding
        elif unique_count <= 30:

            encoder = LabelEncoder()
            feature_df[col] = encoder.fit_transform(feature_df[col].astype(str))

            report.append({
                "Feature": col,
                "Technique": "Label Encoding",
                "Details": f"{unique_count} categories",
            })

        # HIGH CARDINALITY -> Frequency Encoding
        else:

            frequency = feature_df[col].value_counts()
            feature_df[col] = feature_df[col].map(frequency)

            report.append({
                "Feature": col,
                "Technique": "Frequency Encoding",
                "Details": f"{unique_count} categories",
            })

    # --------------------------------------------------------
    # Target Encoding (Only Classification)
    # --------------------------------------------------------

    if target_col is not None and problem_type == "classification" and target is not None:

        temp = feature_df.copy()
        temp[target_col] = target

        for col in categorical_cols:

            if col not in temp.columns:
                continue

            mapping = temp.groupby(col)[target_col].mean()
            temp[col] = temp[col].map(mapping)

            report.append({
                "Feature": col,
                "Technique": "Target Encoding",
                "Details": "Mean target encoding",
            })

        feature_df = temp.drop(columns=[target_col])

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    if target is not None:
        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df, report_df)
