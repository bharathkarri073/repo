# modules/explainability.py
"""
Model explainability: native feature importance, permutation importance,
SHAP global + local explanations. Public API is unchanged —
show_explainability(model, X_train, X_test, y_test) is still the single
entry point — but SHAP explainer construction is now cached
(st.cache_resource) so it doesn't recompute on every Streamlit rerun,
and each section renders inside a themed card instead of a bare
st.subheader.

NOTE: as of the current streamlit_app.py, this module is imported but
show_explainability() is never actually called from any tab — there's
no "Explainability" tab in the tab_overview...tab_problem list. This
file is ready to be wired in as soon as that tab is added; nothing
here depends on that happening first.
"""

import streamlit as st
import pandas as pd
import numpy as np

import shap
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance


def _card_open(title, caption=None):
    caption_html = f'<div class="section-caption">{caption}</div>' if caption else ""
    st.markdown(
        f"""
        <div class="dashboard-card">
            <div class="section-title">{title}</div>
            {caption_html}
        """,
        unsafe_allow_html=True,
    )


def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)


# ===================================
# Feature Importance
# ===================================

def feature_importance(model, X):

    if not hasattr(model, "feature_importances_"):
        st.info("Feature Importance not available for this model.")
        return

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    _card_open("⭐ Feature Importance", "Native importance scores reported by the trained model.")

    st.dataframe(importance, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        importance["Feature"][:15],
        importance["Importance"][:15]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    _card_close()


# ============================================
# Permutation Importance
# ============================================

@st.cache_data(show_spinner=False)
def _compute_permutation_importance(_model, X_test, y_test):
    """
    Cached wrapper around sklearn's permutation_importance. Leading
    underscore on _model tells st.cache_data not to try hashing the
    fitted estimator itself (it isn't hashable in a stable way);
    X_test / y_test are hashed instead, which is enough to invalidate
    the cache whenever the underlying data actually changes.
    """

    return permutation_importance(
        _model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42
    )


def permutation_importance_plot(model, X_test, y_test):

    _card_open("🔄 Permutation Importance", "How much each feature matters when its values are shuffled.")

    result = _compute_permutation_importance(model, X_test, y_test)

    perm = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": result.importances_mean
    })

    perm = perm.sort_values(
        by="Importance",
        ascending=False
    )

    st.dataframe(perm, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.barh(
        perm["Feature"][:15],
        perm["Importance"][:15]
    )

    ax.invert_yaxis()

    st.pyplot(fig)

    _card_close()


# ===================================
# Top Important Features
# ===================================

def top_features(model, X):

    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    _card_open("🏆 Top Important Features")

    st.table(importance.head(10))

    _card_close()


# =====================================
# SHAP Summary
# =====================================

@st.cache_resource(show_spinner="Building SHAP explainer...")
def _get_shap_explainer(_model):
    """
    SHAP explainer construction is the most expensive step in this
    module (can take seconds to tens of seconds on tree ensembles with
    many features). Cached as a resource, keyed off Streamlit's default
    hashing of everything except the leading-underscore _model arg, so
    it's rebuilt once per session per model rather than on every rerun.
    """

    return shap.Explainer(_model)


@st.cache_data(show_spinner="Computing SHAP values...")
def _compute_shap_values(_model, X):
    explainer = _get_shap_explainer(_model)
    return explainer(X)


def shap_summary(model, X):

    _card_open("🧠 SHAP Explainability", "Global feature impact across the evaluation set.")

    try:

        shap_values = _compute_shap_values(model, X)

        fig = plt.figure()

        shap.summary_plot(
            shap_values,
            X,
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning(f"SHAP not supported.\n{e}")

    _card_close()


# =======================================
# Individual Prediction Explanation
# =======================================

def prediction_explanation(model, X):

    _card_open("🔍 Prediction Explanation", "Inspect why the model made a specific prediction.")

    row = st.number_input(
        "Select Row",
        min_value=0,
        max_value=len(X) - 1,
        value=0
    )

    try:

        shap_values = _compute_shap_values(model, X)

        fig = plt.figure()

        shap.plots.waterfall(
            shap_values[row],
            show=False
        )

        st.pyplot(fig)

    except Exception as e:

        st.warning(e)

    _card_close()


# ================================
# Main Function
# ================================

def show_explainability(
    model,
    X_train,
    X_test,
    y_test
):

    st.header("🧠 Model Explainability")

    feature_importance(model, X_train)

    permutation_importance_plot(
        model,
        X_test,
        y_test
    )

    top_features(
        model,
        X_train
    )

    shap_summary(
        model,
        X_test
    )

    prediction_explanation(
        model,
        X_test
    )
