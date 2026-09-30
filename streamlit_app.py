import streamlit as st
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

from modules.datetime_utils import auto_detect_datetime
from modules.preprocessing import intelligent_preprocessing

from modules.visualization import (
    show_data_overview,
    show_unique_values,
    show_missing_analysis,
    numeric_analysis,
    boxplot_analysis,
    categorical_analysis,
    correlation_heatmap,
    multicollinearity_summary,
    time_series_analysis,
    scatter_plot_matrix,
    advanced_scatter_plot
)

from modules.feature_engineering import intelligent_feature_engineering

from modules.automl import (train_automl, get_model_bytes, prediction_csv)

from modules.llm_insights import show_llm_insights

from modules.explainability import show_explainability

from modules.custom_css import apply_professional_theme

from modules.sidebar_ui import (
    render_sidebar_brand,
    render_workflow_status,
    render_sidebar_footer
)

from modules.dashboard import (
    render_hero,
    render_dashboard_kpis,
    render_empty_state,
    render_footer
)

# ---------- Page config (must be first) ----------
st.set_page_config(
    page_title="Gen AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "GenAI Data Scientist Assistant — automated analytics, "
            "AutoML, explainability and AI business intelligence."
        )
    }
)

apply_professional_theme()

# NOTE: render_hero() previously ran once here AND again inside the
# "if uploaded_file is not None" branch below, so the hero banner
# rendered twice on every run once a file was loaded. It's now called
# exactly once, right here, regardless of upload state.
render_hero()


# ---------- Helper Functions ----------

@st.cache_data(ttl=3600)
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding='latin1')
    return df


def detect_target_and_problem(df):

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # ---------------------------------------------------
    # 1. EXCLUDE IDENTIFIER COLUMNS
    # ---------------------------------------------------

    ignore_keywords = [
        'id',
        'index',
        'ticker',
        'symbol',
        'code',
        'name'
    ]

    valid_columns = [
        col for col in df.columns
        if not any(word in col.lower() for word in ignore_keywords)
    ]

    # ---------------------------------------------------
    # 2. STOCK MARKET / TIME SERIES TARGET DETECTION
    # ---------------------------------------------------

    stock_targets = [
        'close',
        'adj close',
        'adjusted close',
        'price',
        'last price'
    ]

    for col in valid_columns:

        if col.lower() in stock_targets:

            return col, 'regression'

    # ---------------------------------------------------
    # 3. CLASSIFICATION TARGETS
    # ---------------------------------------------------

    classification_keywords = [
        'target',
        'label',
        'class',
        'churn',
        'fraud',
        'default',
        'survived'
    ]

    for col in valid_columns:

        if col.lower() in classification_keywords:

            return col, 'classification'

    # ---------------------------------------------------
    # 4. REGRESSION TARGETS
    # ---------------------------------------------------

    regression_keywords = [
        'sales',
        'revenue',
        'profit',
        'income',
        'salary',
        'amount',
        'score'
    ]

    for col in valid_columns:

        if col.lower() in regression_keywords:

            return col, 'regression'

    # ---------------------------------------------------
    # 5. FALLBACK LOGIC
    # ---------------------------------------------------

    candidate_cols = []

    for col in valid_columns:

        unique_ratio = df[col].nunique() / len(df)

        # ignore almost unique columns
        if unique_ratio > 0.95:
            continue

        candidate_cols.append(col)

    # Prefer numeric columns
    numeric_candidates = [
        col for col in candidate_cols
        if col in numeric_cols
    ]

    if numeric_candidates:

        target_col = numeric_candidates[-1]

        if df[target_col].nunique() <= 10:
            return target_col, 'classification'
        else:
            return target_col, 'regression'

    # Final fallback
    target_col = valid_columns[-1]

    if df[target_col].dtype == 'object':
        return target_col, 'classification'

    return target_col, 'regression'


def clean_data_sidebar(df):
    st.sidebar.header("🧹 Data Cleaning")
    with st.sidebar.expander("🗑️ Drop Columns"):
        cols_to_drop = st.multiselect("Select columns to drop", df.columns)
        if st.button("Drop Selected Columns"):
            df.drop(columns=cols_to_drop, inplace=True)
            st.session_state.df = df
            st.rerun()
    with st.sidebar.expander("🔄 Change Data Type"):
        col = st.selectbox("Column", df.columns)
        new_type = st.selectbox("New type", ["int", "float", "str", "datetime", "category", "bool"])
        if st.button("Convert Type"):
            try:
                if new_type == "datetime":
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif new_type == "int":
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                elif new_type == "float":
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                elif new_type == "str":
                    df[col] = df[col].astype(str)
                elif new_type == "category":
                    df[col] = df[col].astype("category")
                elif new_type == "bool":
                    df[col] = df[col].astype(bool)
                st.session_state.df = df
                st.sidebar.success(f"Converted {col} to {new_type}")
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))
    with st.sidebar.expander("👥 Duplicate Rows"):
        dup_count = df.duplicated().sum()
        st.write(f"**Duplicate rows:** {dup_count}")
        if dup_count > 0 and st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.sidebar.success("Duplicates removed!")
            st.rerun()
    return df


def run_automl_with_progress(automl_df, target_col):
    """
    Wraps train_automl() with a live "training model X of N" indicator,
    using the progress_callback hook added to modules/automl.py. Kept
    as its own function so the AutoML tab body below stays readable.
    """

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    def _on_model_trained(model_name, index, total):
        status_placeholder.markdown(
            f"**Training model {index}/{total}:** `{model_name}` ✅"
        )
        progress_bar.progress(index / total)

    results = train_automl(
        automl_df,
        target_col,
        progress_callback=_on_model_trained,
    )

    progress_bar.progress(1.0)
    status_placeholder.markdown("**Training complete.**")

    return results


# ---------- Main App ----------

with st.sidebar:

    render_sidebar_brand()

    st.markdown("### 📁 Data Source")

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        key="main_csv_uploader",
        help="Upload a CSV file for automated analysis."
    )

if uploaded_file is not None:

    df = load_csv(uploaded_file)

    # Auto-convert datetime columns (improved)
    df, datetime_report = auto_detect_datetime(df)

    if uploaded_file.name != st.session_state.get("uploaded_name", ""):
        st.session_state.uploaded_name = uploaded_file.name
        st.session_state.df = df.copy()

    st.session_state.df = clean_data_sidebar(st.session_state.df)

    if "df" not in st.session_state:
        st.session_state.df = df.copy()

    df_clean = st.session_state.df

    with st.sidebar:
        render_workflow_status()

    suggested_target, auto_problem = detect_target_and_problem(df_clean)
    st.sidebar.subheader("🎯 Target Column Setup")
    target_option = st.sidebar.radio("Target selection", ["Auto-detect", "Choose manually"])

    if target_option == "Auto-detect":
        target_col = suggested_target
        problem_type = auto_problem
    else:
        target_col = st.sidebar.selectbox("Select target column", df_clean.columns)
        if df_clean[target_col].dtype in ['object', 'category', 'bool'] or df_clean[target_col].nunique() <= 10:
            problem_type = "classification"
        else:
            problem_type = "regression"

    st.sidebar.info(f"**Detected Problem:** {problem_type.upper()} | **Target:** {target_col}")

    render_dashboard_kpis(
        df=df_clean,
        target_col=target_col,
        problem_type=problem_type
    )

    (
        tab_overview,
        tab_unique,
        tab_missing,
        tab_viz,
        tab_preprocess,
        tab_feature_engineering,
        tab_automl,
        tab_explainability,
        tab_llm,
        tab_problem,
    ) = st.tabs([
        "📄 Data Overview",
        "🔍 Unique Values",
        "❓ Missing Values",
        "📊 Visualizations",
        "🧠 Intelligent Preprocessing",
        "⚙️ Feature Engineering",
        "🤖 AutoML",
        "🔎 Explainability",
        "🧠 Gen AI Insights",
        "📝 Problem Statement",
    ])

    with tab_overview:
        show_data_overview(df_clean)

    with tab_unique:
        show_unique_values(df_clean)

    with tab_missing:
        show_missing_analysis(df_clean)

    with tab_viz:
        st.header("Exploratory Data Analysis")
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        datetime_cols = df_clean.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()
        st.subheader("🕒 Datetime Detection Report")
        st.dataframe(datetime_report, use_container_width=True)
        st.subheader("📈 Numeric Columns Analysis")
        numeric_analysis(df_clean, numeric_cols)
        boxplot_analysis(df_clean, numeric_cols, key_suffix="original")
        st.subheader("📊 Categorical Columns Analysis")
        categorical_analysis(df_clean, categorical_cols, target_col, problem_type)
        if datetime_cols:
            time_series_analysis(df_clean, datetime_cols, numeric_cols)
        st.subheader("📉 Advanced Scatter Plots")
        advanced_scatter_plot(df_clean, numeric_cols, categorical_cols)
        scatter_plot_matrix(df_clean, numeric_cols)
        st.subheader("🔗 Correlation & Multicollinearity")
        correlation_heatmap(df_clean, numeric_cols)
        multicollinearity_summary(df_clean, numeric_cols)
        st.markdown("---")
        st.caption("ℹ️ Use the side panel to clean data further. Next: Missing imputation, AutoML, and RAG+LLM.")

    with tab_preprocess:
        st.title("⚙️ Advanced Data Processing Pipeline")
        if st.button("Run Intelligent Preprocessing"):
            with st.spinner("Processing dataset intelligently..."):
                (processed_df, recommendations_df, imputation_df, skewness_df, outlier_df) = intelligent_preprocessing(df_clean)
                st.session_state.df = processed_df.copy()
                st.session_state["preprocessing_done"] = True
                df_clean = st.session_state.df
                st.subheader("📌 Missing Value Recommendations")
                st.dataframe(recommendations_df, use_container_width=True)
                st.subheader("✅ Missing Value Imputation Status")
                st.dataframe(imputation_df, use_container_width=True)
                processed_numeric_cols = processed_df.select_dtypes(include=np.number).columns.tolist()
                boxplot_analysis(processed_df, processed_numeric_cols, key_suffix="processed")
                st.subheader("📈 Distribution & Skewness Analysis")
                st.dataframe(skewness_df, use_container_width=True)
                st.subheader("🚨 Outlier Detection & Treatment")
                st.dataframe(outlier_df, use_container_width=True)
                st.success("✅ Intelligent preprocessing completed successfully.")

    with tab_feature_engineering:
        st.title("🛠️ Intelligent Feature Engineering")

        scaling_method = st.selectbox("Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
        apply_pca = st.checkbox("Apply PCA", value=False)
        variance_threshold = st.slider("PCA Explained Variance", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

        if st.button("Run Feature Engineering"):
            with st.spinner("Running intelligent feature engineering..."):
                engineered_df, ranking_df, vif_df, engineering_report = intelligent_feature_engineering(
                    df=st.session_state.df,
                    problem_type=problem_type,
                    apply_pca=apply_pca,
                    variance_threshold=variance_threshold,
                    scaling_method=scaling_method,
                    target_col=target_col
                )

            st.session_state["engineered_df"] = engineered_df.copy()

            df_clean = st.session_state.df

            st.success("✅ Intelligent Feature Engineering Completed Successfully")

            st.subheader("📌 Engineered Dataset")
            st.dataframe(engineered_df.head(20), use_container_width=True)

            csv = engineered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Feature Engineered Dataset",
                data=csv,
                file_name="feature_engineered_dataset.csv",
                mime="text/csv"
            )

            st.subheader("🏆 Feature Ranking")
            st.dataframe(ranking_df, use_container_width=True)

            st.subheader("📊 VIF Analysis")
            st.dataframe(vif_df, use_container_width=True)

            st.subheader("📋 Feature Engineering Report")
            st.dataframe(engineering_report, use_container_width=True)

            st.info(
                f"""

                Final Dataset Shape:

                    Rows : {engineered_df.shape[0]}

                    Columns : {engineered_df.shape[1]}

                    """
            )

    with tab_automl:

        st.title("🤖 Intelligent AutoML")

        st.write("Train multiple machine learning models automatically and select the best performing model.")

        if "engineered_df" not in st.session_state:
            st.warning("⚠️ Please run Feature Engineering before AutoML.")
        else:

            if st.button("🚀 Run AutoML"):

                automl_df = st.session_state["engineered_df"]
                st.success(f"✅ Using Feature Engineered Dataset ({automl_df.shape[0]} rows * {automl_df.shape[1]} columns)")

                results = run_automl_with_progress(automl_df, target_col)

                st.session_state["automl_results"] = results
                st.dataframe(results["results"])

                st.plotly_chart(
                    results["visuals"]["comparison"],
                    use_container_width=True
                )

                st.success(
                    f"✅ AutoML Completed Successfully in {results['training_seconds']:.1f}s"
                )

            results = st.session_state.get("automl_results", None)

            if results is not None:

                st.subheader("🏆 Best Model")
                st.success(results["best_model_name"])

                st.subheader("📋 Model Comparison")
                st.dataframe(results["results"], use_container_width=True)

                st.subheader("📊 Model Comparison")
                st.plotly_chart(
                    results["visuals"]["comparison"],
                    use_container_width=True,
                    key="automl_model_comparison"
                )

                if results["visuals"]["importance"] is not None:
                    st.subheader("⭐ Feature Importance")
                    st.pyplot(results["visuals"]["importance"])

                if results["problem_type"] == "classification":

                    st.subheader("📌 Confusion Matrix")
                    st.pyplot(results["visuals"]["confusion"])

                    if results["visuals"]["roc"] is not None:
                        st.subheader("ROC Curve")
                        st.pyplot(results["visuals"]["roc"])

                else:

                    st.subheader("Residual Plot")
                    st.plotly_chart(
                        results["visuals"]["residual"],
                        use_container_width=True,
                        key="automl_residual_plot"
                    )

                    st.subheader("Actual vs Predicted")
                    st.plotly_chart(
                        results["visuals"]["actual"],
                        use_container_width=True,
                        key="automl_actual_predicted"
                    )

                st.subheader("Prediction Results")
                st.dataframe(results["prediction_df"])

                model_buffer = get_model_bytes(results["best_model"])
                st.download_button(
                    "⬇ Download Best Model",
                    data=model_buffer,
                    file_name="best_model.pkl"
                )

                csv = prediction_csv(results["prediction_df"])
                st.download_button(
                    "⬇ Download Prediction CSV",
                    data=csv,
                    file_name="predictions.csv"
                )

    with tab_explainability:

        st.title("🔎 Model Explainability")

        results = st.session_state.get("automl_results", None)

        if results is None:
            st.warning("⚠️ Please run AutoML first to train a model to explain.")
        else:
            show_explainability(
                model=results["best_model"],
                X_train=results["X_train"],
                X_test=results["X_test"],
                y_test=results["y_test"],
            )

    with tab_llm:

        st.title("🧠 Gen AI Business Insights")

        if "automl_results" not in st.session_state:

            st.warning("⚠️ Please complete AutoML first.")

        else:

            problem_statement = st.session_state.get(
                "problem_statement",
                "No business problem statement provided."
            )

            show_llm_insights(
                df=st.session_state.df,
                results=st.session_state["automl_results"],
                problem_statement=problem_statement
            )

    with tab_problem:
        st.subheader("Business Problem Statement")
        problem_text = st.text_area(
            "Paste or type the problem description here:",
            height=150,
            placeholder="Example: Predict customer churn based on usage patterns...",
            help="This will be used later for RAG and LLM analysis."
        )
        if problem_text:
            st.success("Problem statement saved.")
            st.session_state.problem_statement = problem_text
        else:
            st.info("Please enter your problem statement.")

    with st.sidebar:
        render_sidebar_footer()

else:

    render_empty_state()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info(
            "🧹 **Intelligent preprocessing**\n\n"
            "Missing values, outliers and data-quality treatment."
        )

    with c2:
        st.info(
            "⚙️ **Automated engineering**\n\n"
            "Interactions, polynomial features, ranking and VIF."
        )

    with c3:
        st.info(
            "🤖 **AutoML intelligence**\n\n"
            "Model comparison, selection and prediction."
        )

    with c4:
        st.info(
            "🧠 **GenAI insights**\n\n"
            "Executive interpretation and conversational analytics."
        )

    with st.sidebar:
        render_sidebar_footer()

render_footer()
