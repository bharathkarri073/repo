# modules/custom_css.py
"""
Enterprise theme system for the GenAI Data Analyst platform.

Public API is unchanged: call apply_professional_theme() once, right after
st.set_page_config(). Everything else in the app can keep working exactly
as before — this only adds/organizes CSS classes, it doesn't remove any
of the ones already in use (kpi-card, hero-*, dashboard-card, app-footer,
etc. all still exist with the same names).
"""

import streamlit as st


def apply_professional_theme():
    """
    Applies the global visual theme for the GenAI Data Analyst platform.
    Organized into sections: tokens, base, layout, sidebar, hero, kpi,
    cards, nav/tabs, inputs, buttons, dataframes, chat, reports, alerts,
    animations, responsive.
    """

    st.markdown(
        """
        <style>

        /* ==========================================================
           1. DESIGN TOKENS
        ========================================================== */
        @import url(
            'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap'
        );

        :root {
            --primary: #38bdf8;
            --primary-dark: #0369a1;
            --secondary: #8b5cf6;
            --accent: #22d3ee;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;

            --background: #020617;
            --surface: #0f172a;
            --surface-light: #172033;
            --surface-hover: #1e293b;

            --text: #f8fafc;
            --text-secondary: #cbd5e1;
            --muted: #94a3b8;
            --muted-dim: #64748b;

            --border: rgba(148, 163, 184, 0.16);
            --border-strong: rgba(148, 163, 184, 0.28);

            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 20px;
            --radius-xl: 26px;

            --shadow-sm: 0 8px 24px rgba(0, 0, 0, 0.20);
            --shadow-md: 0 18px 45px rgba(0, 0, 0, 0.27);
            --shadow-lg: 0 25px 65px rgba(0, 0, 0, 0.42);

            --font-sans: "Inter", sans-serif;
            --font-mono: "JetBrains Mono", monospace;
        }

        /* ==========================================================
           2. BASE / RESET
        ========================================================== */
        * {
            font-family: var(--font-sans);
        }

        code, pre, kbd, .stCodeBlock, .stCode {
            font-family: var(--font-mono) !important;
        }

        html, body,
        [data-testid="stAppViewContainer"],
        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(56, 189, 248, 0.09), transparent 28%),
                radial-gradient(circle at 88% 16%, rgba(139, 92, 246, 0.10), transparent 30%),
                linear-gradient(145deg, #020617 0%, #07111f 52%, #020617 100%);
            color: var(--text);
        }

        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }

        header[data-testid="stHeader"] {
            background: rgba(2, 6, 23, 0.76);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--border);
        }

        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: var(--background); }
        ::-webkit-scrollbar-thumb {
            background: var(--surface-hover);
            border-radius: 999px;
            border: 2px solid var(--background);
        }
        ::-webkit-scrollbar-thumb:hover { background: var(--muted-dim); }

        /* ==========================================================
           3. LAYOUT
        ========================================================== */
        .block-container {
            max-width: 1500px;
            padding-top: 1.5rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3, h4 {
            color: var(--text);
            letter-spacing: -0.025em;
        }
        h1 { font-weight: 800; }
        h2, h3 { font-weight: 700; }

        /* ==========================================================
           4. SIDEBAR
        ========================================================== */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(8, 20, 38, 0.98), rgba(2, 6, 23, 0.98));
            border-right: 1px solid rgba(56, 189, 248, 0.15);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p { color: var(--text-secondary); }

        /* Sidebar nav items */
        .nav-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            margin-bottom: 4px;
            border-radius: var(--radius-sm);
            color: var(--muted);
            font-size: 0.86rem;
            font-weight: 600;
            border: 1px solid transparent;
            transition: background 0.18s ease, color 0.18s ease, border-color 0.18s ease;
            cursor: pointer;
        }
        .nav-item:hover {
            background: rgba(56, 189, 248, 0.08);
            color: var(--text);
            border-color: rgba(56, 189, 248, 0.18);
        }
        .nav-item.active {
            color: #ffffff;
            background: linear-gradient(100deg, rgba(3, 105, 161, 0.85), rgba(91, 33, 182, 0.80));
            border-color: rgba(125, 211, 252, 0.30);
            box-shadow: 0 6px 18px rgba(37, 99, 235, 0.20);
        }
        .nav-section-label {
            margin: 16px 0 6px 4px;
            color: var(--muted-dim);
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        /* Pipeline status pill */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.68rem;
            font-weight: 700;
        }
        .status-pill.done { color: #86efac; background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.20); }
        .status-pill.pending { color: var(--muted); background: rgba(148,163,184,0.08); border: 1px solid rgba(148,163,184,0.16); }

        /* ==========================================================
           5. HERO
        ========================================================== */
        .ai-hero {
            position: relative;
            overflow: hidden;
            padding: 30px 34px;
            margin-bottom: 24px;
            border-radius: var(--radius-xl);
            background: linear-gradient(125deg, rgba(14, 116, 144, 0.26), rgba(30, 41, 59, 0.88) 48%, rgba(91, 33, 182, 0.23));
            border: 1px solid rgba(125, 211, 252, 0.18);
            box-shadow: var(--shadow-lg), inset 0 1px rgba(255, 255, 255, 0.06);
        }
        .ai-hero::before {
            content: ""; position: absolute; width: 300px; height: 300px;
            top: -180px; right: -90px; border-radius: 50%;
            background: rgba(56, 189, 248, 0.14); filter: blur(5px);
        }
        .ai-hero::after {
            content: ""; position: absolute; width: 230px; height: 230px;
            bottom: -160px; left: 36%; border-radius: 50%;
            background: rgba(139, 92, 246, 0.15);
        }
        .hero-grid {
            display: grid; grid-template-columns: 1fr auto;
            align-items: center; gap: 30px; position: relative; z-index: 2;
        }
        .hero-eyebrow {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 7px 12px; border-radius: 999px; color: #bae6fd;
            background: rgba(14, 165, 233, 0.12); border: 1px solid rgba(56, 189, 248, 0.24);
            font-size: 0.80rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.09em;
        }
        .hero-title {
            margin: 15px 0 9px; font-size: clamp(2rem, 4vw, 3.65rem); line-height: 1.04; font-weight: 800;
            background: linear-gradient(90deg, #f8fafc 0%, #7dd3fc 48%, #c4b5fd 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hero-subtitle { max-width: 800px; color: var(--text-secondary); font-size: 1.02rem; line-height: 1.75; }
        .hero-badges { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; }
        .hero-badge {
            padding: 8px 12px; border-radius: 10px; background: rgba(15, 23, 42, 0.60);
            color: #dbeafe; border: 1px solid var(--border); font-size: 0.82rem; font-weight: 600;
        }
        .robot-orb {
            width: 156px; height: 156px; display: flex; align-items: center; justify-content: center;
            border-radius: 34px; font-size: 78px;
            background: radial-gradient(circle at 32% 28%, rgba(255,255,255,0.22), rgba(56,189,248,0.17) 32%, rgba(139,92,246,0.15) 68%, rgba(15,23,42,0.85));
            border: 1px solid rgba(125, 211, 252, 0.26);
            box-shadow: 0 20px 60px rgba(56, 189, 248, 0.17), inset 0 1px rgba(255, 255, 255, 0.10);
            animation: robotFloat 4s ease-in-out infinite;
        }
        @keyframes robotFloat { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }

        /* ==========================================================
           6. KPI CARDS
        ========================================================== */
        .kpi-card {
            min-height: 142px; padding: 20px; margin-bottom: 12px; border-radius: 19px;
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.92), rgba(8, 18, 34, 0.94));
            border: 1px solid rgba(148, 163, 184, 0.13);
            box-shadow: var(--shadow-md);
            transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            border-color: rgba(56, 189, 248, 0.38);
            box-shadow: 0 23px 55px rgba(14, 165, 233, 0.12);
        }
        .kpi-top { display: flex; align-items: center; justify-content: space-between; }
        .kpi-icon {
            width: 43px; height: 43px; display: flex; align-items: center; justify-content: center;
            border-radius: 13px; font-size: 22px; background: rgba(56, 189, 248, 0.11);
            border: 1px solid rgba(125, 211, 252, 0.17);
        }
        .kpi-status { color: #86efac; font-size: 0.74rem; font-weight: 700; }
        .kpi-value { margin-top: 13px; color: var(--text); font-size: 1.75rem; font-weight: 800; line-height: 1.15; }
        .kpi-label { margin-top: 6px; color: var(--muted); font-size: 0.82rem; font-weight: 500; }

        /* Mini KPI variant (denser rows, e.g. 6-8 per row) */
        .kpi-mini {
            padding: 14px 16px; border-radius: var(--radius-md);
            background: rgba(15, 23, 42, 0.70); border: 1px solid var(--border);
        }
        .kpi-mini .kpi-mini-value { font-size: 1.25rem; font-weight: 800; color: var(--text); }
        .kpi-mini .kpi-mini-label { font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

        /* ==========================================================
           7. SECTION / GENERIC CARDS
        ========================================================== */
        .dashboard-card {
            padding: 22px; margin-bottom: 18px; border-radius: var(--radius-lg);
            background: rgba(15, 23, 42, 0.76); border: 1px solid rgba(148, 163, 184, 0.13);
            box-shadow: var(--shadow-sm); backdrop-filter: blur(13px);
        }
        .section-title { margin-bottom: 4px; color: var(--text); font-size: 1.15rem; font-weight: 750; }
        .section-caption { margin-bottom: 15px; color: var(--muted); font-size: 0.84rem; }

        /* Report / insight cards */
        .report-card {
            padding: 20px 22px; margin-bottom: 14px; border-radius: var(--radius-lg);
            background: rgba(15, 23, 42, 0.80); border-left: 4px solid var(--primary);
            border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
        }
        .report-card.risk { border-left-color: var(--danger); }
        .report-card.recommendation { border-left-color: var(--success); }
        .report-card.insight { border-left-color: var(--accent); }
        .report-card .report-card-title {
            font-weight: 700; font-size: 0.95rem; color: var(--text); margin-bottom: 6px;
            display: flex; align-items: center; gap: 8px;
        }
        .report-card .report-card-body { color: var(--text-secondary); font-size: 0.88rem; line-height: 1.6; }

        .confidence-bar-track {
            width: 100%; height: 6px; border-radius: 999px; background: rgba(148,163,184,0.15); overflow: hidden; margin-top: 8px;
        }
        .confidence-bar-fill {
            height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, var(--primary), var(--secondary));
        }

        /* ==========================================================
           8. INPUTS
        ========================================================== */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"] {
            color: var(--text);
            background: rgba(15, 23, 42, 0.88);
            border-color: rgba(148, 163, 184, 0.18);
            border-radius: 12px;
        }
        textarea {
            color: var(--text) !important;
            background: rgba(15, 23, 42, 0.90) !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 14px !important;
        }

        /* ==========================================================
           9. BUTTONS
        ========================================================== */
        .stButton > button,
        .stDownloadButton > button {
            min-height: 44px; color: white; font-weight: 700;
            border: 1px solid rgba(125, 211, 252, 0.17); border-radius: 12px;
            background: linear-gradient(100deg, #0369a1, #2563eb 48%, #7c3aed);
            box-shadow: 0 10px 27px rgba(37, 99, 235, 0.19);
            transition: transform 0.22s ease, box-shadow 0.22s ease, filter 0.22s ease;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 36px rgba(56, 189, 248, 0.22);
            filter: brightness(1.08);
            color: white;
        }

        /* ==========================================================
           10. TABS
        ========================================================== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 7px; padding: 6px; margin-bottom: 18px;
            border: 1px solid rgba(148, 163, 184, 0.11); border-radius: 15px;
            background: rgba(8, 18, 34, 0.73); overflow-x: auto;
        }
        .stTabs [data-baseweb="tab"] {
            height: auto; min-width: max-content; padding: 10px 13px;
            color: var(--muted); font-size: 0.76rem; font-weight: 650; border-radius: 10px;
        }
        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(100deg, rgba(3, 105, 161, 0.95), rgba(91, 33, 182, 0.92)) !important;
            box-shadow: 0 7px 20px rgba(37, 99, 235, 0.21);
        }

        /* ==========================================================
           11. DATAFRAMES / ALERTS
        ========================================================== */
        [data-testid="stDataFrame"] {
            overflow: hidden; border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 16px; box-shadow: 0 12px 38px rgba(0, 0, 0, 0.20);
        }
        [data-testid="stAlert"] {
            border-radius: 14px; border: 1px solid rgba(148, 163, 184, 0.15);
        }

        /* ==========================================================
           12. CHAT
        ========================================================== */
        [data-testid="stChatMessage"] {
            padding: 14px 16px; margin-bottom: 10px;
            border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 16px;
            background: rgba(15, 23, 42, 0.80);
        }
        .chat-suggestion {
            display: inline-block; padding: 8px 14px; margin: 4px 6px 4px 0;
            border-radius: 999px; background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.20); color: #bae6fd;
            font-size: 0.78rem; font-weight: 600; cursor: pointer;
            transition: background 0.18s ease, border-color 0.18s ease;
        }
        .chat-suggestion:hover { background: rgba(56, 189, 248, 0.16); border-color: rgba(56, 189, 248, 0.36); }

        .typing-dots span {
            display: inline-block; width: 6px; height: 6px; margin-right: 3px;
            border-radius: 50%; background: var(--muted); animation: typingBlink 1.2s infinite ease-in-out;
        }
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes typingBlink { 0%, 80%, 100% { opacity: 0.25; } 40% { opacity: 1; } }

        /* ==========================================================
           13. FOOTER
        ========================================================== */
        .app-footer {
            padding: 24px 10px 4px; color: var(--muted-dim); text-align: center; font-size: 0.78rem;
        }
        .app-footer strong { color: var(--muted); }

        /* ==========================================================
           14. RESPONSIVE
        ========================================================== */
        @media (max-width: 900px) {
            .hero-grid { grid-template-columns: 1fr; }
            .robot-orb { display: none; }
            .ai-hero { padding: 24px; }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
