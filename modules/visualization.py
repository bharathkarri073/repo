import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import gaussian_kde
from statistics import mode as get_mode



def show_data_overview(df):
    st.subheader("📌 Basic Information")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", df.isnull().sum().sum())
    with st.expander("🔍 Preview (first 5 rows)"):
        st.dataframe(df.head(20), use_container_width=True)
    with st.expander("📐 Column Data Types"):
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Column", "Data Type"]
        st.dataframe(dtypes_df)
    with st.expander("📈 Statistical Summary"):
        st.dataframe(df.describe(include='all').T)

def show_unique_values(df):
    st.subheader("🔍 Unique Values Analysis")
    unique_data = []
    for col in df.columns:
        unique_count = len(df[col].unique())
        unique_data.append({"Column": col, "Unique Values Count": unique_count})
    unique_df = pd.DataFrame(unique_data)
    st.dataframe(unique_df, use_container_width=True)
    with st.expander("📋 View Unique Values for a Column"):
        selected_col = st.selectbox("Select column", df.columns, key="unique_values_selectbox")
        unique_values = pd.Series(df[selected_col].unique())
        display_values = unique_values.astype(str)
        display_values = display_values.replace(["nan", "NaT", "None"], "NaN")
        st.write(f"### Unique values in '{selected_col}'")
        st.write(display_values.tolist())

def show_missing_analysis(df):
    st.subheader("❓ Missing Value Analysis")
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_count,
        "Missing Percentage": missing_pct
    })
    missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Percentage", ascending=False)
    if not missing_df.empty:
        st.dataframe(missing_df, use_container_width=True)
        fig = px.bar(missing_df, x="Column", y="Missing Percentage",
                     title="Missing Percentage per Column",
                     color_discrete_sequence=["#7cb9e8"],
                     text="Missing Percentage")
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside', marker_color="#4a90d9")
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="missing_bar")
        st.subheader("Missing Pattern Heatmap")
        missing_matrix = df.isnull().astype(int)
        fig_miss = px.imshow(missing_matrix.T, aspect="auto",
                             color_continuous_scale=["#0a0f1a", "#7cb9e8"],
                             title="Light Blue = Missing")
        fig_miss.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig_miss, use_container_width=True, key="missing_heatmap")
        st.info("💡 Use visualisations below to decide dropping vs imputation.")
    else:
        st.success("✅ No missing values found!")

def numeric_analysis(df, numeric_cols):
    if not numeric_cols:
        st.warning("No numeric columns.")
        return
    selected = st.multiselect("Select numeric columns for detailed histograms", numeric_cols, default=numeric_cols[:min(3, len(numeric_cols))], key="numeric_histogram_multiselect")
    for idx, col in enumerate(selected):
        data = df[col].dropna()
        if len(data) == 0:
            continue
        mean_val = data.mean()
        median_val = data.median()
        try:
            mode_val = get_mode(data)
        except:
            mode_val = data.mode().iloc[0] if not data.mode().empty else np.nan
        x_range = []
        kde_scaled = []
        if len(data) > 1 and data.std() != 0:
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                kde_vals = kde(x_range)
                hist_counts, _ = np.histogram(data, bins=30)
                max_hist = hist_counts.max()
                kde_scaled = (kde_vals / kde_vals.max()) * max_hist
            except:
                pass
        counts, bin_edges = np.histogram(data, bins=20)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bar_width = np.diff(bin_edges)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bin_centers, y=counts, width=bar_width * 0.9, name='Data', opacity=0.85,
                             marker=dict(color='#4a90d9', line=dict(color='white', width=1))))
        fig.add_trace(go.Scatter(x=x_range, y=kde_scaled, mode='lines',
                                 name='Density Curve', line=dict(color='#ffb347', width=3)))
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#00ffcc", annotation_text=f"Mean: {mean_val:.2f}")
        fig.add_vline(x=median_val, line_dash="dash", line_color="#ff66cc", annotation_text=f"Median: {median_val:.2f}")
        fig.add_vline(x=mode_val, line_dash="dash", line_color="#ffff00", annotation_text=f"Mode: {mode_val:.2f}")
        fig.update_layout(barmode='overlay', bargap=0.08, bargroupgap=0.02,
                          title=f"Distribution of {col} (with KDE & statistics)",
                          xaxis_title=col, yaxis_title="Frequency",
                          plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"),
                          legend=dict(font=dict(color="#c0d4f0")))
        st.plotly_chart(fig, use_container_width=True, key=f"hist_{col}_{idx}")

def boxplot_analysis(df, numeric_cols, key_suffix=""):
    st.subheader("📦 Box Plot Analysis (Outlier Visualization)")
    if not numeric_cols:
        st.warning("No numeric columns available.")
        return
    selected_cols = st.multiselect(
        "Select columns for box plot analysis",
        numeric_cols,
        default=numeric_cols[:min(4, len(numeric_cols))],
        key=f"boxplot_multiselect_{key_suffix}"
    )
    for idx, col in enumerate(selected_cols):
        fig = px.box(df, y=col, title=f"Box Plot - {col}", color_discrete_sequence=["#7cb9e8"])
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig, use_container_width=True, key=f"box_{col}_{key_suffix}_{idx}")

def categorical_analysis(df, categorical_cols, target_col=None, problem_type=None):
    if not categorical_cols:
        st.warning("No categorical columns.")
        return
    selected = st.multiselect("Select categorical columns for count plots", categorical_cols,
                              default=categorical_cols[:min(3, len(categorical_cols))],
                              key="categorical_multiselect")
    for idx, col in enumerate(selected):
        freq = df[col].value_counts().reset_index()
        freq.columns = [col, 'count']
        fig = px.bar(freq, x=col, y='count', title=f"Count Plot: {col}",
                     text='count', color=col, color_discrete_sequence=px.colors.sequential.Blues_r)
        fig.update_traces(textposition='outside', marker_line_width=0)
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                          font=dict(family="Times New Roman", color="#c0d4f0"))
        st.plotly_chart(fig, use_container_width=True, key=f"cat_bar_{col}_{idx}")
        if problem_type == 'classification' and target_col and target_col != col and target_col in df.columns:
            st.subheader(f"📊 {col} grouped by Target: {target_col}")
            grouped = df.groupby([target_col, col]).size().reset_index(name='count')
            fig2 = px.bar(grouped, x=col, y='count', color=target_col, text='count',
                          title=f"{col} distribution by {target_col}",
                          barmode='group', color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_traces(textposition='outside')
            fig2.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a",
                               font=dict(family="Times New Roman", color="#c0d4f0"))
            st.plotly_chart(fig2, use_container_width=True, key=f"cat_group_{col}_{idx}")

def correlation_heatmap(df, numeric_cols):
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].dropna(axis=1, how='all').corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                        title="Correlation Matrix (Numeric Columns)")
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
    else:
        st.warning("Need at least 2 numeric columns for correlation.")

def multicollinearity_summary(df, numeric_cols):
    st.subheader("📌 Multicollinearity Analysis Summary")
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric columns.")
        return
    corr_matrix = df[numeric_cols].corr().abs()
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            corr_value = corr_matrix.iloc[i, j]
            if corr_value > 0.80:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr.append({
                    "Column 1": col1,
                    "Column 2": col2,
                    "Correlation": round(corr_value, 2),
                    "Issue": "Multicollinearity Detected",
                    "Recommended Solution": "Consider dropping one column or use PCA/VIF."
                })
    if high_corr:
        st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
    else:
        st.success("✅ No significant multicollinearity detected.")

def time_series_analysis(df, datetime_cols, numeric_cols):
    if not datetime_cols:
        st.info("No datetime columns available for time series analysis.")
        return
    st.subheader("📈 Time Series Analysis")
    selected_time = st.selectbox("Select datetime column", datetime_cols)
    if numeric_cols:
        selected_metrics = st.multiselect("Select numeric columns to plot over time", numeric_cols, default=numeric_cols[:2])
        if selected_metrics:
            df_sorted = df.sort_values(selected_time)
            for idx, metric in enumerate(selected_metrics):
                fig = px.line(df_sorted, x=selected_time, y=metric, title=f"{metric} over time",
                              color_discrete_sequence=["#7cb9e8"])
                fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
                st.plotly_chart(fig, use_container_width=True, key=f"ts_line_{metric}_{idx}")
                window = st.slider(f"Rolling window for {metric}", min_value=2, max_value=min(30, len(df_sorted)), value=7, key=f"roll_{metric}_{idx}")
                df_sorted[f'{metric}_rolling'] = df_sorted[metric].rolling(window=window).mean()
                fig_roll = px.line(df_sorted, x=selected_time, y=[metric, f'{metric}_rolling'],
                                   title=f"{metric} with {window}-period rolling average",
                                   color_discrete_sequence=["#7cb9e8", "#ffb347"])
                fig_roll.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
                st.plotly_chart(fig_roll, use_container_width=True, key=f"ts_roll_{metric}_{idx}")

def scatter_plot_matrix(df, numeric_cols):
    if len(numeric_cols) >= 2:
        st.subheader("🔍 Scatter Plot Matrix (Pair Plot)")
        selected_numeric = st.multiselect("Select up to 5 numeric columns for pair plot", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        if len(selected_numeric) >= 2:
            fig = px.scatter_matrix(df, dimensions=selected_numeric, title="Pair Plot",
                                    color_discrete_sequence=["#7cb9e8"])
            fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
            st.plotly_chart(fig, use_container_width=True, key="scatter_matrix")
    else:
        st.warning("Need at least 2 numeric columns for scatter plot matrix.")

def advanced_scatter_plot(df, numeric_cols, categorical_cols=None):
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for scatter plot.")
        return
    st.subheader("🎯 Interactive Scatter Plot")
    col_x = st.selectbox("X-axis", numeric_cols, index=0)
    col_y = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
    color_col = st.selectbox("Color by (optional)", ["None"] + (categorical_cols if categorical_cols else []))
    plot_type = st.radio("Plot type", ["2D Scatter", "3D Scatter"], horizontal=True)
    if plot_type == "2D Scatter":
        if color_col != "None":
            fig = px.scatter(df, x=col_x, y=col_y, color=color_col, title=f"{col_y} vs {col_x}",
                             color_discrete_sequence=px.colors.qualitative.Set2)
        else:
            fig = px.scatter(df, x=col_x, y=col_y, title=f"{col_y} vs {col_x}",
                             color_discrete_sequence=["#7cb9e8"])
        fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
        st.plotly_chart(fig, use_container_width=True, key="scatter_2d")
    else:
        if len(numeric_cols) >= 3:
            col_z = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1))
            if color_col != "None":
                fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z, color=color_col,
                                    title=f"3D Scatter: {col_x}, {col_y}, {col_z}")
            else:
                fig = px.scatter_3d(df, x=col_x, y=col_y, z=col_z, title=f"3D Scatter: {col_x}, {col_y}, {col_z}",
                                    color_discrete_sequence=["#7cb9e8"])
            fig.update_layout(plot_bgcolor="#0a0f1a", paper_bgcolor="#0a0f1a", font_color="#c0d4f0")
            st.plotly_chart(fig, use_container_width=True, key="scatter_3d")
        else:
            st.warning("Need at least 3 numeric columns for 3D scatter plot.")