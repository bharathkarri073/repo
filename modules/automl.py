import time

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

try:
    from xgboost import XGBRegressor, XGBClassifier
    xgb_available = True
except:
    xgb_available = False

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.metrics import RocCurveDisplay

from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix)

import joblib

from io import BytesIO

# -------------------------------
# Validate Dataset
# -------------------------------

def validate_dataset(df, target_col):

    if df.empty:
        raise ValueError("Dataset is empty.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")

    if len(df) < 20:
        raise ValueError(
    "Dataset contains very few rows for Machine Learning."
)

    return True


# -------------------------------
# Detect Problem Type
# -------------------------------

def detect_problem_type(y):

    if y.dtype == "object":

        return "classification"

    if y.nunique() <= 10:

        return "classification"

    return "regression"


# -------------------------------
# Prepare Dataset
# -------------------------------

def prepare_dataset(df, target_col):

    df = df.copy()

    X = df.drop(columns=[target_col])

    y = df[target_col]

    # AutoML uses only model-ready numeric features.
    X = X.select_dtypes(include=np.number)

    return X, y


# -------------------------------
# Train-Test Split
# -------------------------------

def split_dataset(X, y):

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        shuffle=True

    )

    return (

    X_train,

    X_test,

    y_train,

    y_test

   )


# -------------------------------
# Main Foundation Function
# -------------------------------

def automl_foundation(df, target_col):

    validate_dataset(

        df,

        target_col

    )

    X, y = prepare_dataset(

        df,

        target_col

    )

    problem_type = detect_problem_type(

        y

    )

    X_train, X_test, y_train, y_test = split_dataset(

        X,

        y

    )

    return (

    X_train,

    X_test,

    y_train,

    y_test,

    problem_type,

    X.columns

    )

def regression_automl(X_train, X_test, y_train, y_test, progress_callback=None):
    """
    Runs multiple regression algorithms,
    compares them and returns the best model.

    progress_callback, if provided, is called after each model finishes
    training with (model_name, index, total) so the UI can render a
    live training-progress indicator without this module knowing
    anything about Streamlit widgets.
    """

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(random_state=42),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )

    }

    if xgb_available:

        models["XGBoost"] = XGBRegressor(
            random_state=42,
            n_estimators=200,
            verbosity=0
        )

    results = []

    best_model = None
    best_model_name = None
    best_r2 = -999

    predictions_dict = {}

    total_models = len(models)

    # ---------------------------------------
    # Train Every Model
    # ---------------------------------------

    for idx, (model_name, model) in enumerate(models.items(), start=1):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        predictions_dict[model_name] = predictions

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        results.append({

            "Model": model_name,

            "MAE": round(mae,4),

            "RMSE": round(rmse,4),

            "R² Score": round(r2,4)

        })

        if r2 > best_r2:

            best_r2 = r2

            best_model = model

            best_model_name = model_name

        if progress_callback:
            progress_callback(model_name, idx, total_models)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R² Score",
        ascending=False
    ).reset_index(drop=True)

    return (

        best_model,

        best_model_name,

        results_df,

        predictions_dict

    )



# ===================================================
# Classification AutoML
# ===================================================

def classification_automl(

    X_train,

    X_test,

    y_train,

    y_test,

    progress_callback=None

):

    models = {

        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "Decision Tree":

            DecisionTreeClassifier(random_state=42),

        "Random Forest":

            RandomForestClassifier(random_state=42),

        "Naive Bayes":

            GaussianNB(),

        "KNN":

            KNeighborsClassifier(),

        "SVM":

            SVC(probability=True)

    }

    # Optional XGBoost

    if xgb_available:

        models["XGBoost"] = XGBClassifier(

            eval_metric="logloss",

            random_state=42

        )

    results = []

    predictions_dict = {}

    best_model = None

    best_model_name = ""

    best_accuracy = 0

    total_models = len(models)

    for idx, (name, model) in enumerate(models.items(), start=1):

        model.fit(

            X_train,

            y_train

        )

        predictions = model.predict(

            X_test

        )

        predictions_dict[name] = predictions

        accuracy = accuracy_score(

            y_test,

            predictions

        )

        precision = precision_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        recall = recall_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        f1 = f1_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0

        )

        # ROC-AUC

        try:

            probabilities = model.predict_proba(X_test)

            roc_auc = roc_auc_score(

                y_test,

                probabilities,

                multi_class="ovr"

            )

        except:

            roc_auc = np.nan

        results.append({

            "Model": name,

            "Accuracy": round(accuracy,4),

            "Precision": round(precision,4),

            "Recall": round(recall,4),

            "F1 Score": round(f1,4),

            "ROC-AUC": round(roc_auc,4) if not np.isnan(roc_auc) else "NA"

        })

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model = model

            best_model_name = name

        if progress_callback:
            progress_callback(name, idx, total_models)

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(

        by="Accuracy",

        ascending=False

    ).reset_index(drop=True)

    return (

        best_model,

        best_model_name,

        results_df,

        predictions_dict

    )


# =====================================================
# Model Comparison Chart
# =====================================================

def plot_model_comparison(results_df, problem_type):

    metric = "Accuracy" if problem_type == "classification" else "R² Score"

    chart_df = results_df.sort_values(by=metric, ascending=True)

    fig = px.bar(
        chart_df,
        x=metric,
        y="Model",
        orientation="h",
        text=metric,
        title="Model Leaderboard",
        color=metric,
        color_continuous_scale="Blues"
    )

    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        coloraxis_showscale=False
    )

    return fig


# =====================================================
# Confusion Matrix
# =====================================================

def plot_confusion_matrix(model, X_test, y_test):

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        ax=ax,
        cmap="Blues"
    )

    ax.set_title("Confusion Matrix")

    plt.tight_layout()

    return fig


# =====================================================
# ROC Curve
# =====================================================

def plot_roc_curve(model, X_test, y_test):

    try:

        fig, ax = plt.subplots(figsize=(6, 5))

        RocCurveDisplay.from_estimator(
            model,
            X_test,
            y_test,
            ax=ax
        )

        ax.set_title("ROC Curve")

        plt.tight_layout()

        return fig

    except Exception:

        return None


# =====================================================
# Feature Importance
# =====================================================

def plot_feature_importance(model, feature_names, top_n=15):

    if not hasattr(model, "feature_importances_"):
        return None

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    ).head(top_n)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.barh(

        importance["Feature"],

        importance["Importance"]

    )

    ax.invert_yaxis()

    ax.set_title(

        "Top Feature Importance"

    )

    plt.tight_layout()

    return fig

# =====================================================
# Residual Plot
# =====================================================

def plot_residuals(
    y_test,
    predictions
):

    actual = np.asarray(y_test)
    predicted = np.asarray(predictions)

    residuals = actual - predicted

    residual_df = pd.DataFrame({
        "Predicted": predicted,
        "Residual": residuals,
        "Absolute Error": np.abs(residuals)
    })

    fig = px.scatter(
        residual_df,
        x="Predicted",
        y="Residual",
        color="Absolute Error",
        title="Residual Diagnostics",
        color_continuous_scale="Turbo",
        hover_data=[
            "Absolute Error"
        ]
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)"
    )

    return fig

# =====================================================
# Actual vs Predicted
# =====================================================

def plot_actual_vs_predicted(
    y_test,
    predictions
):

    actual = np.asarray(y_test)
    predicted = np.asarray(predictions)

    plot_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted
    })

    min_value = min(
        actual.min(),
        predicted.min()
    )

    max_value = max(
        actual.max(),
        predicted.max()
    )

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        title="Actual vs Predicted Values",
        opacity=0.78,
        trendline="ols"
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line=dict(
                dash="dash",
                width=2
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)"
    )

    return fig
# =====================================================
# Prediction Distribution
# =====================================================

def plot_prediction_distribution(

    predictions

):

    fig, ax = plt.subplots(figsize=(7,4))

    ax.hist(

        predictions,

        bins=20

    )

    ax.set_title(

        "Prediction Distribution"

    )

    plt.tight_layout()

    return fig

# =====================================================
# Visualization Wrapper
# =====================================================

def generate_visualizations(

    best_model,

    X_test,

    y_test,

    predictions,

    feature_names,

    results_df,

    problem_type

):

    visuals = {}

    visuals["comparison"] = plot_model_comparison(

        results_df,

        problem_type

    )

    visuals["importance"] = plot_feature_importance(

        best_model,

        feature_names

    )

    visuals["distribution"] = plot_prediction_distribution(

        predictions

    )

    if problem_type == "classification":

        visuals["confusion"] = plot_confusion_matrix(

            best_model,

            X_test,

            y_test

        )

        visuals["roc"] = plot_roc_curve(

            best_model,

            X_test,

            y_test

        )

    else:

        visuals["residual"] = plot_residuals(

            y_test,

            predictions

        )

        visuals["actual"] = plot_actual_vs_predicted(

            y_test,

            predictions

        )

    return visuals


# ==========================================================
# MASTER AUTOML PIPELINE
# ==========================================================

def run_automl(
        df,

        target_col,

        progress_callback=None

):

    (
        X_train,
        X_test,
        y_train,
        y_test,
        problem_type,
        feature_names

    ) = automl_foundation(

        df,

        target_col

    )

    start_time = time.time()

    # -----------------------------
    # REGRESSION
    # -----------------------------

    if problem_type == "regression":

        (

            best_model,
            best_model_name,
            results_df,
            predictions_dict

        ) = regression_automl(

            X_train,
            X_test,
            y_train,
            y_test,
            progress_callback=progress_callback

        )

        # -----------------------------
        # CLASSIFICATION
        # -----------------------------

    else:

        (

            best_model,
            best_model_name,
            results_df,
            predictions_dict

        ) = classification_automl(

            X_train,
            X_test,
            y_train,
            y_test,
            progress_callback=progress_callback

        )

    training_seconds = time.time() - start_time

    predictions = best_model.predict(X_test)

    visuals = generate_visualizations(

        best_model,

        X_test,

        y_test,

        predictions,

        feature_names,

        results_df,

        problem_type

    )

    return {

        "problem_type": problem_type,

        "best_model": best_model,

        "best_model_name": best_model_name,

        "results": results_df,

        "predictions": predictions,

        "prediction_dict": predictions_dict,

        "visuals": visuals,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "feature_names": feature_names,

        "training_seconds": training_seconds

}

# ==========================================================
# SAVE TRAINED MODEL
# ==========================================================

def save_model(

    model,

    filename="best_model.pkl"

):

    joblib.dump(

        model,

        filename

    )

    return filename

# ==========================================================
# DOWNLOAD MODEL
# ==========================================================

def get_model_bytes(

    model

):

    buffer = BytesIO()

    joblib.dump(

        model,

        buffer

    )

    buffer.seek(0)

    return buffer


# ==========================================================
# CREATE PREDICTION DATAFRAME
# ==========================================================

def prediction_dataframe(

    y_test,

    predictions

):

    prediction_df = pd.DataFrame({

        "Actual":

            y_test,

        "Predicted":

            predictions

    })

    return prediction_df

# ==========================================================
# DOWNLOAD PREDICTIONS
# ==========================================================

def prediction_csv(

    prediction_df

):

    return prediction_df.to_csv(
        index=False

    ).encode("utf-8")

# ==========================================================
# STREAMLIT WRAPPER
# ==========================================================

def train_automl(df, target_col, progress_callback=None):
    """
    Streamlit-facing entry point. Trains all candidate models for the
    detected problem type, builds the prediction dataframe, and stores
    the wall-clock training time in st.session_state so the dashboard's
    "Training time" KPI (modules/dashboard.py) can display it without
    this module importing anything dashboard-specific.

    progress_callback is optional and, if given, is forwarded down to
    regression_automl / classification_automl so a caller can render a
    live "training model X of N" indicator (e.g. Azure-AutoML-style
    progress bar) during the run.
    """

    results = run_automl(df, target_col, progress_callback=progress_callback)

    prediction_df = prediction_dataframe(results["y_test"], results["predictions"])

    results["prediction_df"] = prediction_df

    st.session_state["automl_training_seconds"] = results["training_seconds"]

    return results
