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
from ui_components import render_chat_interface, render_analytics_sidebar, render_admin_panel

st.set_page_config(
    page_title="Voice Command Shopping Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
        /* ── Lock Page Viewport (Still Page, No Jumping) ── */
        html, body, [data-testid="stAppViewContainer"], .main {
            height: 100vh !important;
            max-height: 100vh !important;
            overflow: hidden !important;
        }

        /* Hide deploy toolbar, hamburger menu, and footer completely */
        #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"], [data-testid="stAppDeployButton"] {
            display: none !important;
            visibility: hidden !important;
        }

        /* Keep header container transparent */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: auto !important;
            min-height: 0 !important;
            padding: 0 !important;
            z-index: 999999 !important;
        }

        /* Beautiful Floating Sidebar Toggle Button (Open / Reopen) */
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            position: fixed !important;
            top: 10px !important;
            left: 12px !important;
            z-index: 1000000 !important;
            background: rgba(25, 29, 38, 0.9) !important;
            backdrop-filter: blur(10px) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35) !important;
            padding: 4px 6px !important;
            cursor: pointer !important;
            transition: all 0.2s ease-in-out !important;
        }

        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] button {
            background: transparent !important;
            border: none !important;
            color: #ffffff !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 2px !important;
        }

        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg {
            fill: #ffffff !important;
            color: #ffffff !important;
            width: 18px !important;
            height: 18px !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover,
        [data-testid="stSidebarCollapseButton"]:hover {
            background: rgba(255, 75, 75, 0.25) !important;
            border-color: rgba(255, 75, 75, 0.5) !important;
            transform: scale(1.05) !important;
        }

        [data-testid="stSidebarCollapsedControl"]:hover svg,
        [data-testid="stSidebarCollapseButton"]:hover svg {
            fill: #FF4B4B !important;
            color: #FF4B4B !important;
        }

        /* ── Layout & Container ── */
        .main .block-container {
            padding-top: 0.8rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 50rem !important;
            margin: 0 auto !important;
            height: calc(100vh - 1.2rem) !important;
            display: flex;
            flex-direction: column;
        }

        /* ── Custom Sleek Header ── */
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            flex-shrink: 0;
        }
        .app-title {
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #FFFFFF 0%, #E0E0E0 50%, #90CAF9 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .app-badge {
            font-size: 0.80rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 14px;
            background: rgba(255, 75, 75, 0.15);
            color: #FF4B4B;
            border: 1px solid rgba(255, 75, 75, 0.3);
        }

        /* ── Chat messages ── */
        div[data-testid="stChatMessage"] {
            padding: 0.5rem 0.8rem !important;
            margin-bottom: 0.35rem !important;
            border-radius: 10px !important;
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.05) !important;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
            line-height: 1.55 !important;
            font-size: 0.98rem !important;
            margin-bottom: 0.25rem !important;
        }

        /* Prevent text wrapping issues */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            min-width: 0;
            overflow-wrap: break-word;
            word-break: normal;
        }

        /* ── Chat input ── */
        div[data-testid="stChatInput"] {
            max-width: 52rem;
            margin: 0 auto;
            flex-shrink: 0;
        }

        div[data-testid="stChatInput"] textarea {
            font-size: 0.95rem !important;
        }

        /* ── Perfectly Balanced Sidebar Cards ── */
        section[data-testid="stSidebar"] {
            min-width: 17rem !important;
            max-width: 32rem !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
            padding-bottom: 0.5rem !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            padding: 0.4rem 0.6rem !important;
            margin-bottom: 0.4rem !important;
            border-radius: 8px !important;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.4rem 0 !important;
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

# Sleek Compact Header
st.markdown(
    """
    <div class="app-header">
        <div class="app-title">🛒 Voice Command Shopping Assistant</div>
        <div class="app-badge">🎙️ Voice + NLP Active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_chat_interface(st.container())

with st.sidebar:
    render_analytics_sidebar(None)
    render_admin_panel()