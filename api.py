# api.py
"""
Voice Command Shopping Assistant REST API Backend (FastAPI - Pure Python)
Exposes REST endpoints for chat, grocery catalog queries, and system health checks.
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
    add_item_to_cart,
    get_cart_total,
    get_cart_items_count,
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
    title="Voice Command Shopping Assistant REST API",
    description="Enterprise REST API Backend for Voice Command Shopping Assistant & Supermarket List Manager",
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
    user_input: str = Field(..., example="Add 2 whole milk and bread to my shopping list")
    session_id: Optional[str] = Field(default="default_session", example="sess_123")


class ChatResponse(BaseModel):
    user_input: str
    bot_response: str
    action: str
    interest_score: int
    cart_changed: bool
    cart_items_count: int
    cart_total: float
    latency_ms: float


class ProductItemResponse(BaseModel):
    product_id: str
    name: str
    category: str
    unit: str
    price: float
    dietary_tags: str
    description: str


# ---------------------------------------------------------------------------
# REST Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Returns system status, loaded catalog count, and vector database health."""
    df = st.session_state.get("df")
    item_count = len(df) if df is not None else 0
    return {
        "status": "healthy",
        "service": "Voice Command Shopping Assistant REST API",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "catalog_items_count": item_count,
    }


@app.post("/api/chat", response_model=ChatResponse, tags=["Chatbot"])
async def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main Chatbot Pipeline Endpoint:
    1. Structured Pydantic LLM intent parsing (English, Hindi, multi-item)
    2. Reference & Cart resolution in session memory
    3. Hybrid ChromaDB + BM25 vector search
    4. Conversational LLM response generation + Smart Suggestions
    5. Interest score tracking
    """
    start_time = time.time()
    user_text = request.user_input.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="user_input cannot be empty.")

    llm = st.session_state.llm
    parsed_intent = parse_intent_with_llm(llm, user_text)
    action_type = parsed_intent.action

    resolved_action = update_state_from_user_message(user_text, parsed_intent=parsed_intent)
    cart_changed = resolved_action.get("cart_changed", False)
    search_results = None

    if action_type == "ADD_TO_CART" and not cart_changed:
        items_to_add = parsed_intent.items or [{"item_name": parsed_intent.cleaned_search_query or user_text, "quantity": 1}]
        added_summaries = []
        all_suggestions = []

        for it in items_to_add:
            item_name = it.item_name if hasattr(it, "item_name") else it.get("item_name", "")
            qty = it.quantity if hasattr(it, "quantity") else it.get("quantity", 1)
            s_results = _hybrid_search(item_name, top_k=3, parsed_intent=parsed_intent)
            if s_results and s_results.get("metadatas") and s_results["metadatas"][0]:
                best_match = s_results["metadatas"][0][0]
                res = add_item_to_cart(best_match, quantity=qty)
                added_summaries.append(f"{qty}x {best_match['name']}")
                all_suggestions.extend(res.get("smart_suggestions", []))
                register_shown_items(s_results["metadatas"][0])

        if added_summaries:
            cart_changed = True
            total = get_cart_total()
            bot_reply = f"🛒 Added {', '.join(added_summaries)} to your list! (Total: ${total:.2f})\n\n"
            if all_suggestions:
                sug_names = " or ".join(f"**{s['name']}** (${s['price']:.2f})" for s in all_suggestions[:2])
                bot_reply += f"💡 **Smart Suggestion**: Shoppers often also add {sug_names}."
        else:
            bot_reply = f"Couldn't locate '{user_text}' in our supermarket catalog."
    elif resolved_action.get("confirmation_override"):
        bot_reply = resolved_action["confirmation_override"]
    else:
        search_prompt = parsed_intent.cleaned_search_query or user_text
        search_results = _hybrid_search(search_prompt, top_k=4, parsed_intent=parsed_intent)
        if search_results and search_results.get("metadatas") and search_results["metadatas"][0]:
            register_shown_items(search_results["metadatas"][0])

        context = _build_enhanced_context(user_text, search_results, parsed_intent=parsed_intent)
        memory_ctx = build_memory_context()
        history = st.session_state.get("chat_history", [])[-6:]
        bot_reply = get_ai_response(llm, user_text, history, context, memory_ctx)

    current_score = st.session_state.get("interest_score", 50)
    new_score = calculate_interest_score(
        user_text,
        current_score,
        resolved_action=resolved_action,
        search_shown=bool(search_results),
    )
    st.session_state.interest_score = new_score

    st.session_state.chat_history.append({"role": "user", "content": user_text})
    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

    latency = round((time.time() - start_time) * 1000, 2)

    return ChatResponse(
        user_input=user_text,
        bot_response=bot_reply,
        action=action_type,
        interest_score=new_score,
        cart_changed=cart_changed,
        cart_items_count=get_cart_items_count(),
        cart_total=get_cart_total(),
        latency_ms=latency,
    )


@app.get("/api/catalog", response_model=List[ProductItemResponse], tags=["Catalog"])
async def get_catalog(
    category: Optional[str] = Query(None, description="Filter by Aisle Category (e.g. Dairy & Eggs, Produce, Bakery, Pantry & Staples, Beverages & Snacks)"),
    search: Optional[str] = Query(None, description="Search products by name or keyword"),
) -> List[ProductItemResponse]:
    """Returns supermarket catalog items with optional category filtering and search query."""
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
            ProductItemResponse(
                product_id=str(row["product_id"]),
                name=str(row["name"]),
                category=str(row["category"]),
                unit=str(row["unit"]),
                price=float(row["price"]),
                dietary_tags=str(row["dietary_tags"]),
                description=str(row["description"]),
            )
        )
    return items
