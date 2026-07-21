# api.py
"""
FoodieBot REST API Backend (FastAPI - 100% Pure Python)
Exposes REST endpoints for chat, menu queries, and system health checks.
Automatic Interactive Swagger Documentation available at: http://localhost:8000/docs
"""

import time
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import streamlit as st

from dotenv import load_dotenv
load_dotenv()

# Mock Streamlit session state for backend API execution
class MockSessionState(dict):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)
    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

if not hasattr(st, "session_state") or not isinstance(st.session_state, MockSessionState):
    st.session_state = MockSessionState()

from database import initialize_services
from bot_logic import (
    initialize_llm,
    parse_intent_with_llm,
    get_ai_response,
    calculate_interest_score,
)
from session_memory import (
    initialize_session_memory,
    update_state_from_user_message,
    build_memory_context,
    build_order_confirmation_message,
    register_shown_items,
    sync_shown_items_from_response,
)
from ui_components import _hybrid_search, _build_enhanced_context, CART_MUTATING_ACTIONS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes ChromaDB, BM25, and LLM services on FastAPI server boot."""
    df, collection, embedder, analysis, bm25, records = initialize_services()
    llm = initialize_llm()

    st.session_state.df = df
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
    yield


app = FastAPI(
    title="FoodieBot REST API",
    description="Enterprise REST API Backend for FoodieBot Conversational Restaurant Assistant",
    version="2.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_input: str = Field(..., example="show me some spicy pizza options")
    session_id: Optional[str] = Field(default="default_session", example="sess_123")


class ChatResponse(BaseModel):
    user_input: str
    bot_response: str
    action: str
    interest_score: int
    cart_changed: bool
    cart_items_count: int
    latency_ms: float


class MenuItemResponse(BaseModel):
    product_id: str
    name: str
    category: str
    price: float
    calories: int
    allergens: str


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Returns system status, loaded menu item count, and database health."""
    df = st.session_state.get("df")
    item_count = len(df) if df is not None else 0
    return {
        "status": "healthy",
        "service": "FoodieBot REST API",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "menu_items_count": item_count,
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main Chatbot Pipeline Endpoint:
    1. Structured Pydantic LLM intent parsing
    2. Reference & Cart resolution in session memory
    3. Hybrid ChromaDB + BM25 vector search
    4. Conversational LLM response generation
    5. Interest score tracking
    """
    start_time = time.time()
    user_text = request.user_input.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="user_input cannot be empty.")

    llm = st.session_state.llm
    parsed_intent = parse_intent_with_llm(llm, user_text)
    resolved_action = update_state_from_user_message(user_text, parsed_intent=parsed_intent)
    action_type = resolved_action["action"]

    search_results = None
    if action_type not in CART_MUTATING_ACTIONS:
        search_prompt = parsed_intent.cleaned_search_query or user_text
        search_results = _hybrid_search(search_prompt, parsed_intent=parsed_intent)
        if search_results and search_results.get("metadatas") and search_results["metadatas"][0]:
            register_shown_items(search_results["metadatas"][0])

    if resolved_action["needs_clarification"]:
        bot_reply = resolved_action["clarification_message"]
    elif resolved_action["cart_changed"]:
        bot_reply = build_order_confirmation_message(action=action_type)
        if action_type == "ADD_TO_CART":
            from ui_components import _suggest_pairing
            pairing = _suggest_pairing(resolved_action)
            if pairing:
                bot_reply += pairing["text"]
                register_shown_items([pairing["metadata"]])
    else:
        context = _build_enhanced_context(user_text, search_results)
        memory_ctx = build_memory_context()
        history = st.session_state.get("chat_history", [])[-6:]
        bot_reply = get_ai_response(llm, user_text, history, context, memory_ctx)

    # Sync any item mentioned in LLM text directly to session memory references
    sync_shown_items_from_response(bot_reply)

    # Interest scoring update
    current_score = st.session_state.get("interest_score", 50)
    search_shown = bool(search_results and search_results.get("metadatas"))
    new_score = calculate_interest_score(
        user_text,
        current_score,
        resolved_action=resolved_action,
        search_shown=search_shown,
    )
    st.session_state.interest_score = new_score

    # Save turn to history
    st.session_state.chat_history.append({"role": "user", "content": user_text})
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    cart = st.session_state.session_memory["order"]["selected_items"]
    latency = round((time.time() - start_time) * 1000, 2)

    return ChatResponse(
        user_input=user_text,
        bot_response=bot_reply,
        action=action_type,
        interest_score=new_score,
        cart_changed=resolved_action["cart_changed"],
        cart_items_count=len(cart),
        latency_ms=latency,
    )


@app.get("/api/menu", response_model=List[MenuItemResponse], tags=["Menu"])
async def get_menu(
    category: Optional[str] = Query(None, description="Filter menu by category (e.g. Pizza, Burgers)"),
    search: Optional[str] = Query(None, description="Search items by name or keyword"),
) -> List[MenuItemResponse]:
    """Returns menu items with optional category filtering and search query."""
    df = st.session_state.get("df")
    if df is None:
        raise HTTPException(status_code=500, detail="Database not initialized.")

    filtered_df = df.copy()

    if category:
        filtered_df = filtered_df[filtered_df["category"].str.lower() == category.lower()]

    if search:
        filtered_df = filtered_df[filtered_df["name"].str.contains(search, case=False, na=False)]

    items = []
    for _, row in filtered_df.iterrows():
        items.append(
            MenuItemResponse(
                product_id=str(row["product_id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                price=float(row["price"]),
                calories=int(row["calories"]),
                allergens=str(row["allergens"]),
            )
        )
    return items
