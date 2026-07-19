# app.py
import streamlit as st
from dotenv import load_dotenv
from database import initialize_services
from bot_logic import initialize_llm
from session_memory import initialize_session_memory
from ui_components import render_chat_interface, render_analytics_sidebar, render_admin_panel

st.set_page_config(
    page_title="FoodieBot Live Demo",
    page_icon="🤖",
    layout="wide",
)
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 92rem;
        }

        div[data-testid="stChatMessage"] {
            padding: 0.35rem 0;
        }

        div[data-testid="stChatMessage"] > div {
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.55;
            margin-bottom: 0.25rem;
        }

        div[data-testid="stChatMessage"] [data-testid="stAvatarIcon"] {
            margin-top: 0.15rem;
        }

        div[data-testid="stChatMessage"]:has([data-testid="stAvatarIcon"] svg) > div {
            background: rgba(255, 255, 255, 0.04);
        }

        div[data-testid="stChatMessage"]:has(button[aria-label="user"]) > div {
            background: rgba(92, 111, 255, 0.12);
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

    st.toast("✅ All services initialized!", icon="✅")
    st.rerun()

# Main UI
st.title("🤖 FoodieBot Live Dashboard")
render_chat_interface(st.container())

with st.sidebar:

    render_analytics_sidebar(None)
    render_admin_panel()