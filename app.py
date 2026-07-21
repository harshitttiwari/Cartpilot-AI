# app.py
import streamlit as st
from dotenv import load_dotenv
from database import initialize_services
from bot_logic import initialize_llm
from session_memory import initialize_session_memory
from ui_components import render_chat_interface, render_analytics_sidebar, render_admin_panel

st.set_page_config(
    page_title="Foodie Assistant Bot",
    page_icon="🤖",
    layout="wide",
)
st.markdown(
    """
    <style>
        /* ── Layout & Container ── */
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 0;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 48rem;
            margin: 0 auto;
        }

        /* Hide Streamlit chrome */
        #MainMenu, header[data-testid="stHeader"],
        footer, .stDeployButton {
            display: none !important;
        }

        /* ── Chat messages ── */
        div[data-testid="stChatMessage"] {
            padding: 0.6rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        }

        div[data-testid="stChatMessage"] > div {
            border-radius: 0;
            padding: 0.5rem 0;
            box-shadow: none;
            border: none;
            background: transparent !important;
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.6;
            font-size: 0.95rem;
            margin-bottom: 0.3rem;
        }

        /* Prevent cart-confirmation text from collapsing into
           single-character-per-line vertical text */
        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            min-width: 0;
            overflow-wrap: anywhere;
            word-break: normal;
            white-space: normal;
        }

        /* ── Chat input ── */
        div[data-testid="stChatInput"] {
            max-width: 48rem;
            margin: 0 auto;
        }

        /* ── Scrollable chat area ── */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
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

# Main UI
st.title("🤖 Foodie Assistant Bot")
render_chat_interface(st.container())

with st.sidebar:

    render_analytics_sidebar(None)
    render_admin_panel()