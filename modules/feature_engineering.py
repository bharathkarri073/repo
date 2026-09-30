import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import (LabelEncoder,StandardScaler,MinMaxScaler,PolynomialFeatures)
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression, f_classif
from sklearn.feature_selection import (mutual_info_classif,mutual_info_regression)

from sklearn.ensemble import (RandomForestClassifier,RandomForestRegressor)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
# ============================================================
# PART 1
# INTELLIGENT FEATURE SELECTION
# ============================================================

def intelligent_feature_selection(
    df,
    target_col=None
):

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

        feature_df.drop(
            columns=duplicate_columns,
            inplace=True
        )

        for col in duplicate_columns:

            report.append({

                "Feature": col,

                "Technique": "Duplicate Feature Removal",

                "Details": "Duplicate of another feature"

            })

    # ========================================================
    # 2. Remove Constant Features
    # ========================================================

    constant_features = []

    for col in feature_df.columns:

        if feature_df[col].nunique(dropna=False) <= 1:

            constant_features.append(col)

    if constant_features:

        feature_df.drop(

            columns=constant_features,

            inplace=True

        )

        for col in constant_features:

            report.append({

                "Feature": col,

                "Technique": "Constant Feature Removal",

                "Details": "Only one unique value"

            })

    # ========================================================
    # 3. Remove Near Constant Features
    # ========================================================

    near_constant = []

    threshold = 0.98

    for col in feature_df.columns:

        try:

            top_frequency = (

                feature_df[col]

                .value_counts(normalize=True)

                .iloc[0]

            )

            if top_frequency > threshold:

                near_constant.append(col)

        except Exception:

            pass

    if near_constant:

        feature_df.drop(

            columns=near_constant,

            inplace=True

        )

        for col in near_constant:

            report.append({

                "Feature": col,

                "Technique": "Near Constant Removal",

                "Details": "Dominated by one value"

            })

    # ========================================================
    # 4. Remove Low Variance Numeric Features
    # ========================================================

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

    if len(numeric_cols) > 0:

        selector = VarianceThreshold(

            threshold=0.01

        )

        selector.fit(

            feature_df[numeric_cols]

        )

        selected = [

            c

            for c, keep in zip(

                numeric_cols,

                selector.get_support()

            )

            if keep

        ]

        removed = [
            col
            for col in numeric_cols
            if col not in selected
        ]

        if removed:

            feature_df.drop(

                columns=removed,

                inplace=True

            )

            for col in removed:

                report.append({

                    "Feature": col,

                    "Technique": "Low Variance Removal",

                    "Details": "Variance < 0.01"

                })

    # ========================================================
    # 5. Remove ID Columns
    # ========================================================

    id_keywords = [

        "id",

        "index",

        "customerid",

        "userid",

        "employeeid",

        "transactionid",

        "recordid"

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

        feature_df.drop(

            columns=id_columns,

            inplace=True

        )

        for col in id_columns:

            report.append({

                "Feature": col,

                "Technique": "Identifier Removal",

                "Details": "Unique identifier"

            })

    # ========================================================
    # Restore Target
    # ========================================================

    if target is not None:

        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df,report_df)


# ============================================================
# PART 2
# INTELLIGENT DATETIME FEATURE ENGINEERING
# ============================================================

def datetime_feature_engineering(
    df,
    target_col=None
):

    df = df.copy()

    report = []

    # Find datetime columns only
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

            df[f"{col}_week"] = (
                df[col]
                .dt.isocalendar()
                .week
                .astype(int)
            )

            df[f"{col}_quarter"] = df[col].dt.quarter

            df[f"{col}_is_month_start"] = (
                df[col].dt.is_month_start.astype(int)
            )

            df[f"{col}_is_month_end"] = (
                df[col].dt.is_month_end.astype(int)
            )

            df[f"{col}_is_quarter_start"] = (
                df[col].dt.is_quarter_start.astype(int)
            )

            df[f"{col}_is_quarter_end"] = (
                df[col].dt.is_quarter_end.astype(int)
            )

            df[f"{col}_is_year_start"] = (
                df[col].dt.is_year_start.astype(int)
            )

            df[f"{col}_is_year_end"] = (
                df[col].dt.is_year_end.astype(int)
            )

            df[f"{col}_is_weekend"] = (
                df[col].dt.dayofweek >= 5
            ).astype(int)

            # ---------------------------------------
            # Time Features
            # ---------------------------------------

            df[f"{col}_hour"] = df[col].dt.hour

            df[f"{col}_minute"] = df[col].dt.minute

            df[f"{col}_second"] = df[col].dt.second

            # ---------------------------------------
            # Cyclic Encoding
            # ---------------------------------------

            df[f"{col}_month_sin"] = np.sin(
                2 * np.pi * df[f"{col}_month"] / 12
            )

            df[f"{col}_month_cos"] = np.cos(
                2 * np.pi * df[f"{col}_month"] / 12
            )

            df[f"{col}_dayofweek_sin"] = np.sin(
                2 * np.pi * df[f"{col}_dayofweek"] / 7
            )

            df[f"{col}_dayofweek_cos"] = np.cos(
                2 * np.pi * df[f"{col}_dayofweek"] / 7
            )

            df[f"{col}_hour_sin"] = np.sin(
                2 * np.pi * df[f"{col}_hour"] / 24
            )

            df[f"{col}_hour_cos"] = np.cos(
                2 * np.pi * df[f"{col}_hour"] / 24
            )

            # ---------------------------------------
            # Feature Age
            # ---------------------------------------

            today = pd.Timestamp.today()

            df[f"{col}_days_from_today"] = (
                today - df[col]
            ).dt.days

            report.append({

                "Feature": col,

                "Technique": "Datetime Feature Engineering",

                "Details":
                    "Generated 20+ calendar, cyclic and time features"

            })

        except Exception:

            pass

    # ---------------------------------------
    # Remove Original Datetime Columns
    # ---------------------------------------

    df.drop(

        columns=datetime_cols,

        inplace=True,

        errors="ignore"

    )

    return df, pd.DataFrame(report)
    
# ============================================================
# PART 3
# INTELLIGENT CATEGORICAL ENGINEERING
# ============================================================

def intelligent_categorical_engineering(

    df,

    target_col=None,

    problem_type=None

):

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

        # ==========================================
        # LOW CARDINALITY
        # One-Hot Encoding
        # ==========================================

        if unique_count <= 10:

            onehot = pd.get_dummies(

                feature_df[col],

                prefix=col,

                drop_first=True

            )

            feature_df.drop(

                columns=[col],

                inplace=True

            )

            feature_df = pd.concat(

                [feature_df, onehot],

                axis=1

            )

            report.append({

                "Feature": col,

                "Technique": "One-Hot Encoding",

                "Details": f"{unique_count} categories"

            })

        # ==========================================
        # MEDIUM CARDINALITY
        # Label Encoding
        # ==========================================

        elif unique_count <= 30:

            encoder = LabelEncoder()

            feature_df[col] = encoder.fit_transform(

                feature_df[col].astype(str)

            )

            report.append({

                "Feature": col,

                "Technique": "Label Encoding",

                "Details": f"{unique_count} categories"

            })

        # ==========================================
        # HIGH CARDINALITY
        # Frequency Encoding
        # ==========================================

        else:

            frequency = feature_df[col].value_counts()

            feature_df[col] = feature_df[col].map(frequency)

            report.append({

                "Feature": col,

                "Technique": "Frequency Encoding",

                "Details": f"{unique_count} categories"

            })

    # --------------------------------------------------------
    # Target Encoding
    # (Only Classification)
    # --------------------------------------------------------

    if (target_col is not None
        and
        problem_type == "classification"
        and
        target is not None

    ):

        temp = feature_df.copy()

        temp[target_col] = target

        for col in categorical_cols:

            if col not in temp.columns:

                continue

            mapping = (

                temp

                .groupby(col)[target_col]

                .mean()

            )


            temp[col] = temp[col].map(mapping)

            report.append({

                "Feature": col,

                "Technique": "Target Encoding",

                "Details": "Mean target encoding"

            })

        feature_df = temp.drop(columns=[target_col])

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    if target is not None:

        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df,report_df)
    
# ============================================================
# PART 4
# INTELLIGENT NUMERIC FEATURE ENGINEERING
# ============================================================

def intelligent_numeric_engineering(

    df,

    target_col=None

):

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
    # Numeric Columns
    # --------------------------------------------------------

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

    # --------------------------------------------------------
    # Process Every Numeric Feature
    # --------------------------------------------------------

    for col in numeric_cols:

        try:

            # --------------------------------------------
            # Replace Infinite Values
            # --------------------------------------------

            feature_df[col] = feature_df[col].replace(

                [np.inf, -np.inf],

                np.nan

            )

            feature_df[col] = feature_df[col].fillna(

                feature_df[col].median()

            )

            # --------------------------------------------
            # Skewness
            # --------------------------------------------

            skewness = feature_df[col].skew()

            # --------------------------------------------
            # Highly Skewed
            # Log Transform
            # --------------------------------------------

            if abs(skewness) > 1:

                shift = 0

                if feature_df[col].min() <= 0:

                    shift = abs(feature_df[col].min()) + 1

                feature_df[col] = np.log1p(

                    feature_df[col] + shift

                )

                new_skew = feature_df[col].skew()

                report.append({

                    "Feature": col,

                    "Technique": "Log Transformation",

                    "Details": f"{round(skewness,2)} -> {round(new_skew,2)}"

                })

            # --------------------------------------------
            # Moderately Skewed
            # Square Root
            # --------------------------------------------

            elif abs(skewness) > 0.5:

                shift = 0

                if feature_df[col].min() < 0:

                    shift = abs(feature_df[col].min())

                feature_df[col] = np.sqrt(

                    feature_df[col] + shift

                )

            report.append({

                "Feature": col,

                "Technique": "Square Root Transformation",

                "Details": f"Skewness={round(skewness,2)}"

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

            feature_df[col] = feature_df[col].clip(

                lower,

                upper

            )

            changed = (

                original != feature_df[col]

            ).sum()

            if changed > 0:

                report.append({

                    "Feature": col,

                    "Technique": "Outlier Clipping",

                    "Details": f"{changed} values clipped"

                })

        except Exception:

            pass

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    if target is not None:

        feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (feature_df,report_df)


# ============================================================
# PART 5
# SMART INTERACTION FEATURE GENERATION
# ============================================================

def smart_interaction_features(

    df,

    target_col,

    problem_type

):

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

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

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

            scores, _ = f_regression(

                X,

                y

            )

        else:

            scores, _ = f_classif(

                X,

                y

            )

        importance = pd.DataFrame({

            "Feature": numeric_cols,

            "Score": scores

        })

        importance = importance.sort_values(

            by="Score",

            ascending=False

        )
        importance.reset_index(drop = True, inplace = True)

        top_features = importance["Feature"].head(5).tolist()

    except:

        top_features = numeric_cols[:min(5, len(numeric_cols))]

    # --------------------------------------------------------
    # Multiplication Features
    # --------------------------------------------------------

    for i in range(len(top_features)):

        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]

            col2 = top_features[j]

            new_col = f"{col1}_x_{col2}"

            feature_df[new_col] = (

                feature_df[col1]

                *

                feature_df[col2]

            )

            report.append({

                "Feature": new_col,

                "Technique": "Interaction Feature",

                "Details": f"{col1} × {col2}"

            })

    # --------------------------------------------------------
    # Ratio Features
    # --------------------------------------------------------

    for i in range(len(top_features)):

        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]

            col2 = top_features[j]

            new_col = f"{col1}_div_{col2}"

            denominator = feature_df[col2].replace(

                0,

                np.nan

            )

            feature_df[new_col] = (

                feature_df[col1]

                /

                denominator

            )

            feature_df[new_col] = feature_df[new_col].fillna(

                0
            )

            report.append({

                "Feature": new_col,

                "Technique": "Ratio Feature",

                "Details": f"{col1} / {col2}"

            })

    # --------------------------------------------------------
    # Difference Features
    # --------------------------------------------------------

    for i in range(len(top_features)):

        for j in range(i + 1, len(top_features)):

            col1 = top_features[i]

            col2 = top_features[j]

            new_col = f"{col1}_minus_{col2}"

            feature_df[new_col] = (

                feature_df[col1]

                -

                feature_df[col2]

            )

            report.append({

                "Feature": new_col,

                "Technique": "Difference Feature",

                "Details": f"{col1} - {col2}"

            })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (

        feature_df,

        report_df

    )

# ============================================================
# PART 6
# INTELLIGENT POLYNOMIAL FEATURE GENERATION
# ============================================================

def intelligent_polynomial_features(

    df,

    target_col,

    problem_type,

    max_features=5,

    degree=2

):

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

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

    if len(numeric_cols) == 0:

        feature_df[target_col] = target

        return feature_df, pd.DataFrame(report)

    # --------------------------------------------------------
    # Select Top Important Features
    # --------------------------------------------------------

    try:

        X = feature_df[numeric_cols]

        y = target

        if problem_type == "regression":

            scores, _ = f_regression(

                X,

                y

            )

        else:

            scores, _ = f_classif(

                X,

                y

            )

        importance = pd.DataFrame({

            "Feature": numeric_cols,

            "Score": scores

        })

        importance = importance.sort_values(

            by="Score",

            ascending=False

        )
        importance.reset_index(drop = True, inplace = True)

        selected_features = importance[

            "Feature"

        ].head(max_features).tolist()

    except:

        selected_features = numeric_cols[:min(max_features, len(numeric_cols))]

    # --------------------------------------------------------
    # Polynomial Features
    # --------------------------------------------------------

    poly = PolynomialFeatures(

        degree=degree,

        include_bias=False

    )

    transformed = poly.fit_transform(

        feature_df[selected_features]

    )

    poly_names = poly.get_feature_names_out(

        selected_features

    )

    poly_df = pd.DataFrame(

        transformed,

        columns=poly_names,

        index=feature_df.index

    )

    # --------------------------------------------------------
    # Keep ONLY newly generated columns
    # --------------------------------------------------------

    new_columns = [

        col

        for col in poly_df.columns

        if col not in selected_features

    ]
    if len(new_columns) == 0:
        
        feature_df[target_col] = target

        return feature_df, pd.DataFrame(report)

    poly_df = poly_df[new_columns]

    feature_df = pd.concat(

        [

            feature_df,

            poly_df

        ],

        axis=1

    )

    # --------------------------------------------------------
    # Engineering Report
    # --------------------------------------------------------

    for col in new_columns:

        report.append({

            "Feature": col,

            "Technique": "Polynomial Feature",

            "Details": f"Degree {degree}"

        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (

        feature_df,

        report_df
    
    )

# ============================================================
# PART 7
# MUTUAL INFORMATION + RANDOM FOREST FEATURE IMPORTANCE +AUTOMATIC FEATURE SELECTION
# ============================================================

def intelligent_feature_ranking(

    df,

    target_col,

    problem_type,

    top_features=30

):

    df = df.copy()

    report = []

    target = df[target_col]

    feature_df = df.drop(columns=[target_col])

    # --------------------------------------------------------
    # Keep only numeric columns
    # --------------------------------------------------------

    numeric_df = feature_df.select_dtypes(

        include=np.number

    )

    if numeric_df.empty:

        df[target_col] = target

        return (

            df,

            pd.DataFrame(),

            pd.DataFrame()

        )

    X = numeric_df.copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.fillna(0)

    y = target

    # --------------------------------------------------------
    # Mutual Information
    # --------------------------------------------------------

    if problem_type == "classification":

        mi_scores = mutual_info_classif(

            X,

            y,

            random_state=42

        )

        model = RandomForestClassifier(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        )

    else:

        mi_scores = mutual_info_regression(

            X,

            y,

            random_state=42

        )

        model = RandomForestRegressor(

            n_estimators=300,

            random_state=42,

            n_jobs=-1

        )

    # --------------------------------------------------------
    # Random Forest Importance
    # --------------------------------------------------------

    model.fit(

        X,

        y

    )

    rf_scores = model.feature_importances_

    # --------------------------------------------------------
    # Normalize Scores
    # --------------------------------------------------------

    scaler = MinMaxScaler()

    mi_scores = scaler.fit_transform(

        mi_scores.reshape(-1,1)

    ).flatten()

    rf_scores = scaler.fit_transform(

        rf_scores.reshape(-1,1)

    ).flatten()

    # --------------------------------------------------------
    # Combined Score
    # --------------------------------------------------------

    ranking_df = pd.DataFrame({

        "Feature": X.columns,

        "Mutual Information": mi_scores,

        "Random Forest Importance": rf_scores

    })

    ranking_df["Combined Score"] = (

        ranking_df["Mutual Information"]

        +

        ranking_df["Random Forest Importance"]

    ) / 2

    ranking_df = ranking_df.sort_values(

        by="Combined Score",

        ascending=False

    )

    ranking_df.reset_index(drop = True, inplace = True)

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

    removed = sorted([col for col in feature_df.columns
                if col not in selected_features
    ])

    for col in removed:

        report.append({

            "Feature": col,

            "Technique": "Feature Selection",

            "Details": "Removed due to low combined importance"

        })

    report_df = pd.DataFrame(report)

    return (

        final_df,

        ranking_df,

        report_df

    )
    
# ============================================================
# PART 8
# INTELLIGENT MULTICOLLINEARITY REMOVAL
# Correlation + VIF
# ============================================================

def intelligent_multicollinearity_removal(

    df,

    target_col,

    ranking_df,

    corr_threshold=0.90,

    vif_threshold=10

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

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

    if len(numeric_cols) <= 1:

        feature_df[target_col] = target

        return (

            feature_df,

            pd.DataFrame(),

            pd.DataFrame()

        )

    # --------------------------------------------------------
    # Feature Importance Dictionary
    # --------------------------------------------------------

    importance = dict(

        zip(

            ranking_df["Feature"],

            ranking_df["Combined Score"]

        )

    )

    # --------------------------------------------------------
    # Correlation Matrix
    # --------------------------------------------------------

    corr_matrix = (feature_df[numeric_cols]
                   .replace([np.inf, -np.inf], np.nan)
                   .fillna(0)
                   .corr()
                   .abs()
    )

    upper = corr_matrix.where(

        np.triu(

            np.ones(corr_matrix.shape),

            k=1

        ).astype(bool)

    )

    remove_columns = set()

    # --------------------------------------------------------
    # Remove Less Important Correlated Feature
    # --------------------------------------------------------

    for col in upper.columns:

        correlated = upper.index[

            upper[col] > corr_threshold

        ].tolist()

        for other in correlated:

            score_col = importance.get(col, 0)

            score_other = importance.get(other, 0)

            if score_col >= score_other:

                remove_columns.add(other)

            else:

                remove_columns.add(col)

    if remove_columns:

        feature_df.drop(

            columns=list(remove_columns),

            inplace=True,

            errors="ignore"

        )

        for col in remove_columns:

            report.append({

                "Feature": col,

                "Technique": "Correlation Removal",

                "Details": f"Correlation > {corr_threshold}"

            })

    # --------------------------------------------------------
    # VIF Calculation
    # --------------------------------------------------------

    vif_df = pd.DataFrame()
    while True:

        numeric_cols = feature_df.select_dtypes(

            include=np.number

        ).columns.tolist()

        if len(numeric_cols) <= 1:

            break

        vif_df = pd.DataFrame()

        vif_df["Feature"] = numeric_cols

        vif_df["VIF"] = [

            variance_inflation_factor(

                feature_df[numeric_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
                .values,

                i

            )

            for i in range(len(numeric_cols))

        ]
        vif_df = vif_df.sort_values(
            by="VIF",
            ascending=False
        ).reset_index(drop=True)

        max_vif = vif_df["VIF"].max()

        if max_vif <= vif_threshold:

            break

        remove_feature = vif_df.iloc[0]["Feature"]

        feature_df.drop(

            columns=[remove_feature],

            inplace=True

        )

        report.append({

            "Feature": remove_feature,

            "Technique": "VIF Removal",

            "Details": f"VIF={round(max_vif,2)}"

        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (

        feature_df,

        vif_df,

        report_df

    )
    
# ============================================================
# PART 9
# INTELLIGENT SCALING + PCA
# ============================================================

def intelligent_scaling_pca(

    df,

    target_col,

    scaling_method="StandardScaler",

    apply_pca=False,

    variance_threshold=0.95

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

    numeric_cols = feature_df.select_dtypes(

        include=np.number

    ).columns.tolist()

    if len(numeric_cols) == 0:

        if target is not None:
            feature_df[target_col] = target
        report_df = pd.DataFrame(report)

        return (feature_df, report_df)

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
            feature_df[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        feature_df[numeric_cols] = scaler.fit_transform(

            feature_df[numeric_cols]

        )

        report.append({

            "Feature": "Numeric Features",

            "Technique": scaling_method,

            "Details": "Scaling Applied"

        })

    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    if apply_pca and len(numeric_cols) > 1:

        pca = PCA(

            n_components=variance_threshold,

            random_state=42

        )

        transformed = pca.fit_transform(

            feature_df[numeric_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        pca_columns = [

            f"PC{i+1}"

            for i in range(

                transformed.shape[1]

            )

        ]

        pca_df = pd.DataFrame(

            transformed,

            columns=pca_columns,

            index=feature_df.index

        )

        # Drop original numeric features

        feature_df.drop(

            columns=numeric_cols,

            inplace=True

        )

        feature_df = pd.concat(

            [

                feature_df,

                pca_df

            ],

            axis=1

        )

        explained = round(
            np.sum(pca.explained_variance_ratio_) * 100,
            2
        )

        report.append({

            "Feature": "Principal Components",

            "Technique": "PCA",

            "Details":

                f"{len(pca_columns)} PCs retained ({explained}% variance explained)"


        })

    # --------------------------------------------------------
    # Restore Target
    # --------------------------------------------------------

    feature_df[target_col] = target

    report_df = pd.DataFrame(report)

    return (

        feature_df,

        report_df

    )

# ============================================================
# PART 10
# MASTER FEATURE ENGINEERING PIPELINE
# ============================================================

def intelligent_feature_engineering(

    df,

    target_col,

    problem_type,

    scaling_method="StandardScaler",

    apply_pca=False,

    variance_threshold=0.95

):
    df = df.copy()
    engineering_reports = []

    ranking_df = pd.DataFrame()
    vif_df = pd.DataFrame()

    # =====================================================
    # PART 1
    # =====================================================

    df, report = intelligent_feature_selection(

        df,

        target_col

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 2
    # =====================================================

    df, report = datetime_feature_engineering(

        df,

        target_col

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 3
    # =====================================================

    df, report = intelligent_categorical_engineering(

        df,

        target_col,

        problem_type

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 4
    # =====================================================

    df, report = intelligent_numeric_engineering(

        df,

        target_col

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 5
    # =====================================================

    df, report = smart_interaction_features(

        df,

        target_col,

        problem_type

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 6
    # =====================================================

    df, report = intelligent_polynomial_features(

        df,

        target_col,

        problem_type

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 7
    # =====================================================

    df, ranking_df, report = intelligent_feature_ranking(

        df,

        target_col,

        problem_type

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 8
    # =====================================================

    df, vif_df, report = intelligent_multicollinearity_removal(

        df,

        target_col,

        ranking_df

    )

    engineering_reports.append(report)

    # =====================================================
    # PART 9
    # =====================================================

    df, report = intelligent_scaling_pca(

        df,

        target_col,

        scaling_method,

        apply_pca,

        variance_threshold

    )

    engineering_reports.append(report)

    # =====================================================
    # Final Report
    # =====================================================

    engineering_reports = [
        report

        for report in engineering_reports

        if report is not None and not report.empty

    ]

    if engineering_reports:
        final_report = pd.concat(
            engineering_reports,

            ignore_index=True
        )

    else:
        final_report = pd.DataFrame()

    return (

        df,

        ranking_df,

        vif_df,

        final_report

    )