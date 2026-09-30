import time
from io import BytesIO
from html import escape

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = "llama-3.3-70b-versatile"

SUGGESTED_QUESTIONS = [
    "What are the strongest predictors in this dataset?",
    "Are there any data quality issues I should worry about?",
    "How confident should I be in the best model's predictions?",
    "What would you investigate next?",
]

# Section keywords used to color-code parsed report cards.
# Order matters — first match wins.
_SECTION_STYLE_RULES = [
    (("risk",), "risk", "⚠️"),
    (("recommendation", "action"), "recommendation", "✅"),
    (("future", "improvement"), "recommendation", "🚀"),
    (("feature", "importance"), "insight", "⭐"),
    (("model", "performance"), "insight", "🎯"),
    (("dataset", "data insight"), "insight", "📊"),
    (("executive", "summary"), "insight", "🧭"),
]


# ==========================================================
# API Key Handling
# ==========================================================

def _check_api_key():
    """
    Renders the API-key status banner. Previously this ran at import
    time, which meant it rendered on every tab of the app regardless
    of whether the LLM Insights tab was even open. Now it's only
    called from inside show_llm_insights(), so it only shows up where
    it's relevant.
    """

    if API_KEY:
        st.success("✅ Groq API key loaded successfully")
        return True

    st.error("❌ Groq API key not found. Set GROQ_API_KEY in your .env file.")
    return False


# ==========================================================
# Configure Groq
# ==========================================================

def configure_llm(api_key):

    client = Groq(api_key=api_key)

    return client


# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    summary = f"""

Dataset Information

Rows : {df.shape[0]}

Columns : {df.shape[1]}

Numeric Columns :
    {numeric_cols}

Categorical Columns :
    {categorical_cols}

Missing Values :
    {df.isnull().sum().sum()}

Duplicate Rows :
    {df.duplicated().sum()}

"""

    return summary


# ==========================================================
# Model Summary
# ==========================================================

def model_summary(results):

    problem_type = results.get(
        "problem_type",
        "unknown"
    )

    best_model_name = results.get(
        "best_model_name",
        "Unknown"
    )

    results_df = results.get(
        "results",
        pd.DataFrame()
    )

    if results_df.empty:

        metrics_text = "Model metrics unavailable."

    else:

        matching_row = results_df[
            results_df["Model"] == best_model_name
        ]

        if matching_row.empty:
            matching_row = results_df.head(1)

        # NOTE: this line previously only executed inside the
        # "matching_row.empty" branch, which meant metrics_text was
        # never assigned in the normal case (best_model_name found) —
        # an UnboundLocalError waiting to happen. Fixed by moving it
        # outside the empty-check so it always runs.
        metrics_text = matching_row.to_string(index=False)

    return f"""
Problem Type:
{problem_type}

Best Model:
{best_model_name}

Best Model Performance:
{metrics_text}
"""


# ==========================================================
# Feature Importance Summary
# ==========================================================

def feature_summary(results):

    model = results["best_model"]

    feature_names = results["feature_names"]

    if hasattr(model, "feature_importances_"):

        importance = pd.DataFrame({

            "Feature": feature_names,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(

            by="Importance",

            ascending=False

        ).head(10)

        return importance.to_string(index=False)

    return "Feature importance unavailable."


# ==========================================================
# Prompt Builder
# ==========================================================

def build_prompt(

    problem_statement,

    dataset_info,

    model_info,

    feature_info

):

    prompt = f"""

You are an experienced Senior Data Scientist.

Analyze this Machine Learning project.

---------------------------------------------------

Business Problem

{problem_statement}

---------------------------------------------------

Dataset Summary

{dataset_info}

---------------------------------------------------

Model Summary

{model_info}

---------------------------------------------------

Top Important Features

{feature_info}

---------------------------------------------------

Please provide, using these exact section headings prefixed with "## ":

## Executive Summary
## Dataset Insights
## Model Performance Analysis
## Feature Importance Interpretation
## Business Recommendations
## Risks
## Future Improvements

Explain everything in simple business language.

"""

    return prompt


# ==========================================================
# Generate Insights (non-streaming, kept for compatibility)
# ==========================================================

def generate_insights(client, prompt):

    try:

        response = client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0.3

        )

        return response.choices[0].message.content

    except Exception as e:

        st.error(f"Groq Error: {e}")

        return None


# ==========================================================
# Streaming variants (used by the UI for a "live typing" feel)
# ==========================================================

def _stream_groq_text(client, messages):
    """
    Generator yielding text chunks from a Groq chat completion.
    Wrapped so a network/API failure surfaces as a single yielded
    error string instead of raising mid-stream and breaking
    st.write_stream.
    """

    try:

        stream = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    except Exception as e:

        yield f"\n\n_Groq Error: {e}_"


def stream_insights(client, prompt):
    return _stream_groq_text(client, [{"role": "user", "content": prompt}])


def stream_answer(client, dataset_info, question):

    prompt = f"""

Dataset Summary

{dataset_info}

Question

{question}

Answer only based on the dataset summary.

"""

    return _stream_groq_text(client, [{"role": "user", "content": prompt}])


# ==========================================================
# Dataset Chat (non-streaming, kept for compatibility)
# ==========================================================

def ask_question(

    client,

    dataset_info,

    question

):
    try:

        prompt = f"""

Dataset Summary

{dataset_info}

Question

{question}

Answer only based on the dataset summary.

"""

        response = client.chat.completions.create(
            model=GROQ_MODEL,

            messages=[

            {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0.3

        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"


def calculate_dataset_health(df):

    if df is None or df.empty:
        return 0

    total_cells = max(
        df.shape[0] * df.shape[1],
        1
    )

    missing_ratio = (
        df.isna().sum().sum()
        / total_cells
    )

    duplicate_ratio = (
        df.duplicated().sum()
        / max(len(df), 1)
    )

    score = 100 - (
        missing_ratio * 70
        + duplicate_ratio * 30
    ) * 100

    return round(
        max(0, min(100, score)),
        1
    )


def extract_primary_metric(results):

    results_df = results.get(
        "results",
        pd.DataFrame()
    )

    best_model_name = results.get(
        "best_model_name"
    )

    if results_df.empty:
        return "Unavailable", 0.0

    best_row = results_df[
        results_df["Model"] == best_model_name
    ]

    if best_row.empty:
        best_row = results_df.head(1)

    row = best_row.iloc[0]

    if results.get("problem_type") == "classification":

        value = float(
            row.get("Accuracy", 0)
        )

        return "Accuracy", value

    value = float(
        row.get("R² Score", 0)
    )

    return "R² score", value


def create_feature_importance_df(results):

    model = results.get("best_model")

    feature_names = list(
        results.get(
            "feature_names",
            []
        )
    )

    if (
        model is None
        or not hasattr(
            model,
            "feature_importances_"
        )
    ):
        return pd.DataFrame()

    importance_values = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values
    })

    return importance_df.sort_values(
    by="Importance",
    ascending=False
).head(15)


def render_ai_kpi_dashboard(
    df,
    results
):

    health_score = calculate_dataset_health(
        df
    )

    metric_name, metric_value = (
        extract_primary_metric(results)
    )

    best_model = results.get(
        "best_model_name",
        "Unknown"
    )

    confidence = min(
        abs(metric_value) * 100,
        100
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            "Dataset Health",
            f"{health_score}%"
        )

    with k2:
        st.metric(
            "Best Model",
            best_model
        )

    with k3:

        formatted_metric = (
            f"{metric_value * 100:.2f}%"
            if metric_name == "Accuracy"
            else f"{metric_value:.4f}"
        )

        st.metric(
            metric_name,
            formatted_metric
        )

    with k4:
        st.metric(
            "Model Confidence",
            f"{confidence:.1f}%"
        )


def render_model_comparison_chart(
    results
):

    results_df = results.get(
        "results",
        pd.DataFrame()
    ).copy()

    if results_df.empty:
        return

    if (
        results.get("problem_type")
        == "classification"
    ):

        metric = "Accuracy"

    else:

        metric = "R² Score"

    chart_df = results_df.sort_values(
        by=metric,
        ascending=True
    )

    fig = px.bar(
        chart_df,
        x=metric,
        y="Model",
        orientation="h",
        text=metric,
        title="Model Performance Leaderboard",
        hover_data=chart_df.columns
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="ai_model_leaderboard"
    )


def render_feature_importance_chart(
    results
):

    importance_df = (
        create_feature_importance_df(
            results
        )
    )

    if importance_df.empty:

        st.info(
            "Native feature importance is "
            "not available for this model."
        )

        return

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        title="Global Feature Importance",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="ai_feature_importance"
    )


def render_prediction_diagnostics(
    results
):

    if (
        results.get("problem_type")
        != "regression"
    ):
        return

    y_test = results.get("y_test")
    predictions = results.get(
        "predictions"
    )

    if (
        y_test is None
        or predictions is None
    ):
        return

    actual = np.asarray(y_test)
    predicted = np.asarray(predictions)

    diagnostics_df = pd.DataFrame({
        "Actual": actual,
        "Predicted": predicted,
        "Residual": actual - predicted
    })

    fig = px.scatter(
        diagnostics_df,
        x="Actual",
        y="Predicted",
        color="Residual",
        title="Prediction Accuracy Diagnostics",
        color_continuous_scale="RdBu",
        hover_data=["Residual"]
    )

    min_value = min(
        actual.min(),
        predicted.min()
    )

    max_value = max(
        actual.max(),
        predicted.max()
    )

    fig.add_trace(
        go.Scatter(
            x=[min_value, max_value],
            y=[min_value, max_value],
            mode="lines",
            name="Perfect prediction",
            line=dict(
                dash="dash"
            )
        )
    )

    fig.update_layout(
        template="plotly_dark",
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.55)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="ai_prediction_diagnostics"
    )


# ==========================================================
# Report parsing → styled cards
# ==========================================================

def _style_for_heading(heading_text):

    lowered = heading_text.lower()

    for keywords, css_class, icon in _SECTION_STYLE_RULES:
        if any(keyword in lowered for keyword in keywords):
            return css_class, icon

    return "insight", "📄"


def _parse_report_sections(report_text):
    """
    Splits the LLM's markdown report on "## " headings into
    (heading, body) pairs. If the model didn't follow the requested
    heading format, returns a single ("AI Report", full_text) section
    so nothing is silently dropped.
    """

    if not report_text or "## " not in report_text:
        return [("AI Report", report_text or "")]

    sections = []
    current_heading = None
    current_lines = []

    for line in report_text.splitlines():
        if line.strip().startswith("## "):
            if current_heading is not None:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line.strip()[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections if sections else [("AI Report", report_text)]


def render_report_cards(report_text):

    sections = _parse_report_sections(report_text)

    for heading, body in sections:

        css_class, icon = _style_for_heading(heading)

        with st.expander(f"{icon}  {heading}", expanded=True):
            st.markdown(
                f'<div class="report-card {css_class}">'
                f'<div class="report-card-body">',
                unsafe_allow_html=True,
            )
            st.markdown(body if body else "_No content generated for this section._")
            st.markdown("</div></div>", unsafe_allow_html=True)


# ==========================================================
# PDF / DOCX export
# ==========================================================

def build_docx_report(report_text, title="AI Business Report"):
    """
    Returns .docx bytes, or None if python-docx isn't installed.
    Caller is responsible for showing an install hint in that case.
    """

    try:
        from docx import Document
    except ImportError:
        return None

    document = Document()
    document.add_heading(title, level=1)

    for heading, body in _parse_report_sections(report_text):
        document.add_heading(heading, level=2)
        for paragraph in body.split("\n\n"):
            if paragraph.strip():
                document.add_paragraph(paragraph.strip())

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)

    return buffer.getvalue()


def build_pdf_report(report_text, title="AI Business Report"):
    """
    Returns PDF bytes, or None if reportlab isn't installed.
    """

    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 16)]

    for heading, body in _parse_report_sections(report_text):
        story.append(Paragraph(heading, styles["Heading2"]))
        story.append(Spacer(1, 6))
        for paragraph in body.split("\n\n"):
            if paragraph.strip():
                safe_text = escape(paragraph.strip()).replace("\n", "<br/>")
                story.append(Paragraph(safe_text, styles["BodyText"]))
                story.append(Spacer(1, 6))
        story.append(Spacer(1, 12))

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()


def render_export_buttons(report_text):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            "⬇ Download as TXT",
            data=report_text,
            file_name="Business_AI_Report.txt",
            mime="text/plain",
        )

    with col2:
        docx_bytes = build_docx_report(report_text)
        if docx_bytes is not None:
            st.download_button(
                "⬇ Download as DOCX",
                data=docx_bytes,
                file_name="Business_AI_Report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        else:
            st.caption("Install `python-docx` to enable DOCX export.")

    with col3:
        pdf_bytes = build_pdf_report(report_text)
        if pdf_bytes is not None:
            st.download_button(
                "⬇ Download as PDF",
                data=pdf_bytes,
                file_name="Business_AI_Report.pdf",
                mime="application/pdf",
            )
        else:
            st.caption("Install `reportlab` to enable PDF export.")


# ==========================================================
# Chat UI
# ==========================================================

def render_chat_ui(client, dataset_info):

    st.subheader("💬 Ask AI")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
    clicked_suggestion = None

    for col, suggestion in zip(suggestion_cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(suggestion, key=f"suggestion_{suggestion}", use_container_width=True):
                clicked_suggestion = suggestion

    top_row = st.columns([5, 1])

    with top_row[0]:
        question = st.text_input(
            "Ask anything about your project",
            key="chat_question_input",
            label_visibility="collapsed",
            placeholder="Ask anything about your project…",
        )

    with top_row[1]:
        ask_clicked = st.button("Ask AI", key="ask_ai_btn", use_container_width=True)

    if st.button("🗑️ Clear Chat", key="clear_chat_btn"):
        st.session_state["chat_history"] = []
        st.rerun()

    final_question = clicked_suggestion or (question if ask_clicked else None)

    for sender, message in st.session_state["chat_history"]:
        role = "user" if sender == "You" else "assistant"
        with st.chat_message(role):
            st.markdown(message)

    if final_question:

        if not final_question.strip():
            st.warning("Please enter a question.")
            return

        st.session_state["chat_history"].append(("You", final_question))

        with st.chat_message("user"):
            st.markdown(final_question)

        with st.chat_message("assistant"):
            answer = st.write_stream(
                stream_answer(client, dataset_info, final_question)
            )

        st.session_state["chat_history"].append(("AI", answer))


# ==========================================================
# Main Streamlit UI
# ==========================================================

def show_llm_insights(

    df,

    results,

    problem_statement

):

    api_key_ok = _check_api_key()

    if not api_key_ok:
        st.stop()

    client = configure_llm(API_KEY)

    dataset_info = dataset_summary(df)

    model_info = model_summary(results)

    st.markdown("## 📊 AI Executive Dashboard")

    render_ai_kpi_dashboard(df, results)

    chart_left, chart_right = st.columns(2)

    with chart_left:
        render_model_comparison_chart(results)

    with chart_right:
        render_feature_importance_chart(results)

    render_prediction_diagnostics(results)

    st.markdown("---")

    feature_info = feature_summary(results)

    with st.expander("📄 Raw Context Sent to the LLM (dataset / model / feature summaries)"):
        st.text(dataset_info)
        st.text(model_info)
        st.text(feature_info)

    if st.button("✨ Generate AI Insights", key="generate_ai_btn"):

        prompt = build_prompt(
            problem_statement,
            dataset_info,
            model_info,
            feature_info,
        )

        with st.chat_message("assistant"):
            report_text = st.write_stream(stream_insights(client, prompt))

        if report_text:
            st.session_state["report"] = report_text

    st.markdown("## 📑 AI Business Report")

    if "report" in st.session_state:

        render_report_cards(st.session_state["report"])

        render_export_buttons(st.session_state["report"])

    else:

        st.info("Click **Generate AI Insights** above to produce the executive report.")

    st.markdown("---")

    render_chat_ui(client, dataset_info)
