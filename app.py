# app.py
import os
import sys
import logging
import warnings

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

import streamlit as st
from dotenv import load_dotenv
from database import initialize_services
from bot_logic import initialize_llm
from session_memory import initialize_session_memory
import importlib
import ui_components
import voice_component
importlib.reload(ui_components)
importlib.reload(voice_component)
from ui_components import render_chat_interface, render_analytics_sidebar, render_admin_panel

st.set_page_config(
    page_title="Cartpilot Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <!-- v2.6 cache-bust-input80 -->
    <style>
        /* ── Typography (preserves icon fonts in spans) ── */
        html, body, p, input, textarea, button, label,
        h1, h2, h3, h4, h5, h6, a, li, td, th,
        [data-testid="stMarkdownContainer"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        /* ── Viewport Lock & Root Layout ── */
        html, body, [data-testid="stAppViewContainer"], section.main {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }

        /* Expand main content area to fill screen nicely */
        [data-testid="stAppViewBlockContainer"],
        .block-container {
            max-width: 96% !important;
            width: 96% !important;
            margin: 0 auto !important;
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
            padding-left: 1.2rem !important;
            padding-right: 1.2rem !important;
            height: 100vh !important;
            max-height: 100vh !important;
            box-sizing: border-box !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            overflow: hidden !important;
        }

        /* ── Hide Streamlit chrome ── */
        #MainMenu, footer, .stDeployButton, [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            display: none !important;
            height: 0px !important;
        }

        /* ── Sidebar reopen always visible ── */
        [data-testid="stSidebarCollapsedControl"] {
            opacity: 1 !important;
        }

        /* ── Sidebar collapse fix (Streamlit 1.60) ── */
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0px !important;
            min-width: 0px !important;
            overflow: hidden !important;
        }

        /* ═══════════════════════════════════════════ */
        /*        PREMIUM CHAT UI — GPT/CLAUDE STYLE  */
        /* ═══════════════════════════════════════════ */

        /* ── Big Prominent Header ── */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.2rem 0;
            margin-bottom: 0.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            flex-shrink: 0;
        }
        .app-title {
            font-size: 1.65rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 10px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .app-badge {
            font-size: 0.78rem;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 20px;
            background: rgba(255, 75, 75, 0.15);
            color: #FF4B4B;
            border: 1px solid rgba(255, 75, 75, 0.3);
            letter-spacing: 0.02em;
        }

        /* ── Chat Container (scrollable area) ── */
        .block-container div[data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            background: rgba(255, 255, 255, 0.02) !important;
            border-radius: 14px !important;
            height: 450px !important;
            min-height: 450px !important;
            max-height: 450px !important;
            margin-bottom: 0.4rem !important;
        }

        /* The inner scrolling block */
        .block-container div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
            overflow-y: auto !important;
            padding-right: 0.5rem;
        }

        /* ── Chat Messages — Claude/GPT Style ── */
        div[data-testid="stChatMessage"] {
            padding: 0.7rem 0.9rem !important;
            margin-bottom: 0.2rem !important;
            border-radius: 12px !important;
            border: none !important;
            background: transparent !important;
            transition: background 0.2s ease;
        }
        div[data-testid="stChatMessage"]:hover {
            background: rgba(255, 255, 255, 0.03) !important;
        }

        /* Assistant messages */
        div[data-testid="stChatMessage"]:has(.role-marker-assistant) {
            border-left: 2px solid rgba(99, 102, 241, 0.4) !important;
        }

        /* User messages */
        div[data-testid="stChatMessage"]:has(.role-marker-user) {
            background: rgba(99, 102, 241, 0.08) !important;
            border-left: 2px solid rgba(168, 85, 247, 0.4) !important;
        }

        /* Message text */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            line-height: 1.6 !important;
            font-size: 0.93rem !important;
            color: #d1d5db !important;
        }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong {
            color: #ffffff !important;
            font-weight: 600;
        }
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            min-width: 0;
            overflow-wrap: break-word;
            word-break: normal;
        }

        /* ── Single-Line Horizontal Chat Input ── */
        div[data-testid="stChatInput"] {
            padding: 0 !important;
            margin: 0 auto !important;
            width: 80% !important;
            max-width: 80% !important;
        }
        div[data-testid="stChatInput"] > div {
            display: flex !important;
            flex-direction: row !important;
            align-items: center !important;
            justify-content: space-between !important;
            min-height: 42px !important;
            height: 42px !important;
            max-height: 42px !important;
            padding: 4px 10px !important;
            border-radius: 20px !important;
            box-sizing: border-box !important;
            overflow: hidden !important;
        }
        div[data-testid="stChatInput"] > div > div:first-child {
            flex: 1 1 auto !important;
            display: flex !important;
            align-items: center !important;
            width: 100% !important;
        }
        div[data-testid="stChatInput"] textarea,
        textarea[data-testid="stChatInputTextArea"] {
            flex: 1 !important;
            width: 100% !important;
            min-height: 26px !important;
            height: 26px !important;
            max-height: 26px !important;
            padding: 2px 6px !important;
            font-size: 0.90rem !important;
            line-height: 22px !important;
            resize: none !important;
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            margin: 0 !important;
        }
        div[data-testid="stChatInput"] [data-testid="stChatInputInstructions"],
        div[data-testid="stChatInput"] small,
        div[data-testid="stChatInput"] > div > div:nth-child(2):not(:last-child) {
            display: none !important;
        }
        div[data-testid="stChatInput"] > div > div:last-child {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: auto !important;
            height: auto !important;
            margin: 0 0 0 6px !important;
            padding: 0 !important;
            flex-shrink: 0 !important;
        }
        button[data-testid="stChatInputSubmitButton"] {
            height: 30px !important;
            width: 30px !important;
            min-height: 30px !important;
            max-height: 30px !important;
            min-width: 30px !important;
            max-width: 30px !important;
            border-radius: 50% !important;
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            color: white !important;
            margin: 0 !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
        }

        /* ── Sidebar — Refined ── */
        section[data-testid="stSidebar"] {
            min-width: 18rem !important;
            max-width: 28rem !important;
            background: #0d0f14 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 0.6rem 0.8rem !important;
        }
        section[data-testid="stSidebar"] h4 {
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            letter-spacing: 0.02em !important;
            color: #9ca3af !important;
            margin-bottom: 0.4rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.5rem 0.7rem !important;
            margin-bottom: 0.35rem !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.35rem 0 !important;
            border-color: rgba(255, 255, 255, 0.06) !important;
        }

        /* ── Buttons — Polished ── */
        .stButton > button {
            border-radius: 10px !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            padding: 0.35rem 0.8rem !important;
            transition: all 0.2s ease !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            border: none !important;
            color: white !important;
        }

        /* ── Streamlit Widgets Cleanup ── */
        div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.06) !important;
            border-radius: 10px !important;
            background: rgba(255, 255, 255, 0.02) !important;
        }
        .stProgress > div > div {
            background: linear-gradient(90deg, #6366f1, #a78bfa) !important;
            border-radius: 8px !important;
        }
        .stSpinner > div {
            border-top-color: #6366f1 !important;
        }

        /* ── Scrollbar — Minimal ── */
        ::-webkit-scrollbar {
            width: 5px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "app_ready" not in st.session_state:
    with st.spinner("Initializing services..."):
        load_dotenv()
        df, collection, embedder, analysis, bm25, records = initialize_services()
        llm = initialize_llm()

    if df is None:
        st.error("❌ Failed to load product data. Check CSV file.")
        st.stop()
    if collection is None or embedder is None:
        st.error("❌ Failed to initialize vector database or embeddings.")
        st.stop()
    if llm is None:
        st.error("❌ Failed to initialize LLM. Check GEMINI_API_KEY.")
        st.stop()

    st.session_state.df = df
    st.session_state.df_analysis = analysis
    st.session_state.collection = collection
    st.session_state.embedder = embedder
    st.session_state.bm25 = bm25
    st.session_state.records = records
    st.session_state.llm = llm
    st.session_state.chat_history = []
    st.session_state.interest_score = 50
    st.session_state.interest_history = [50]
    st.session_state.query_log = []
    st.session_state.order = {}
    initialize_session_memory()
    st.session_state.app_ready = True

    st.toast("All services initialized!")
    st.rerun()

# Minimal Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🛒 Cartpilot Assistant</div>
        <div class="app-badge">🎙️ Voice + NLP Active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_chat_interface(st.container())

with st.sidebar:
    render_analytics_sidebar(None)
    render_admin_panel()