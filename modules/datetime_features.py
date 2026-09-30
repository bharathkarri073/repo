"""
PART 2 — Intelligent Datetime Feature Engineering

Expands every datetime column into calendar features, cyclic
(sin/cos) encodings, time-of-day features, and a "days from today"
recency feature, then drops the original datetime columns. Logic is
unchanged from the original file — only relocated.
"""

import pandas as pd
import numpy as np


def datetime_feature_engineering(df, target_col=None):

    df = df.copy()
    report = []

    datetime_cols = df.select_dtypes(
        include=["datetime64[ns]", "datetime64"]
    ).columns.tolist()

    if len(datetime_cols) == 0:
        return df, pd.DataFrame(report)

    for col in datetime_cols:

        try:

            # ---------------------------------------
            # Calendar Features
            # ---------------------------------------

            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
            df[f"{col}_dayofyear"] = df[col].dt.dayofyear
            df[f"{col}_week"] = df[col].dt.isocalendar().week.astype(int)
            df[f"{col}_quarter"] = df[col].dt.quarter
            df[f"{col}_is_month_start"] = df[col].dt.is_month_start.astype(int)
            df[f"{col}_is_month_end"] = df[col].dt.is_month_end.astype(int)
            df[f"{col}_is_quarter_start"] = df[col].dt.is_quarter_start.astype(int)
            df[f"{col}_is_quarter_end"] = df[col].dt.is_quarter_end.astype(int)
            df[f"{col}_is_year_start"] = df[col].dt.is_year_start.astype(int)
            df[f"{col}_is_year_end"] = df[col].dt.is_year_end.astype(int)
            df[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)

            # ---------------------------------------
            # Time Features
            # ---------------------------------------

            df[f"{col}_hour"] = df[col].dt.hour
            df[f"{col}_minute"] = df[col].dt.minute
            df[f"{col}_second"] = df[col].dt.second

            # ---------------------------------------
            # Cyclic Encoding
            # ---------------------------------------

            df[f"{col}_month_sin"] = np.sin(2 * np.pi * df[f"{col}_month"] / 12)
            df[f"{col}_month_cos"] = np.cos(2 * np.pi * df[f"{col}_month"] / 12)
            df[f"{col}_dayofweek_sin"] = np.sin(2 * np.pi * df[f"{col}_dayofweek"] / 7)
            df[f"{col}_dayofweek_cos"] = np.cos(2 * np.pi * df[f"{col}_dayofweek"] / 7)
            df[f"{col}_hour_sin"] = np.sin(2 * np.pi * df[f"{col}_hour"] / 24)
            df[f"{col}_hour_cos"] = np.cos(2 * np.pi * df[f"{col}_hour"] / 24)

            # ---------------------------------------
            # Feature Age
            # ---------------------------------------

            today = pd.Timestamp.today()
            df[f"{col}_days_from_today"] = (today - df[col]).dt.days

            report.append({
                "Feature": col,
                "Technique": "Datetime Feature Engineering",
                "Details": "Generated 20+ calendar, cyclic and time features",
            })

        except Exception:
            pass

    # ---------------------------------------
    # Remove Original Datetime Columns
    # ---------------------------------------

    df.drop(columns=datetime_cols, inplace=True, errors="ignore")

    return df, pd.DataFrame(report)
