# bot_logic.py
import os
import re
import json
from types import SimpleNamespace
from typing import Literal, List, Optional
import streamlit as st
from pydantic import BaseModel, Field
from google import genai
from groq import Groq
from log import log_llm_provider, log_llm_failover, log_intent_parsed, log_interest_score


# ---------------------------------------------------------------------------
# LLM Initialization (Gemini with Groq Fallback)
# ---------------------------------------------------------------------------

@st.cache_resource
def initialize_llm():
    """
    Initializes a dual-provider LLM adapter.
    Primary: Gemini (google-genai)
    Fallback: Groq llama-3.3-70b-versatile (on 429 / quota errors)
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    gemini_client = None    
    groq_client = None

    if gemini_key:
        try:
            gemini_client = genai.Client(api_key=gemini_key)
        except Exception as e:
            st.warning(f"⚠️ Gemini init failed: {e}. Will rely on Groq only.")
    else:
        st.warning("⚠️ GEMINI_API_KEY not set — Gemini unavailable.")

    if groq_key:
        try:
            groq_client = Groq(api_key=groq_key)
        except Exception as e:
            st.warning(f"⚠️ Groq init failed: {e}.")
    else:
        st.warning("⚠️ GROQ_API_KEY not set — Groq fallback unavailable.")

    if gemini_client is None and groq_client is None:
        st.error("❌ No LLM provider available. Set GEMINI_API_KEY or GROQ_API_KEY.")
        return None

    return _DualProviderAdapter(
        gemini_client=gemini_client,
        gemini_model=model_name,
        groq_client=groq_client,
        groq_model=groq_model,
    )


class _DualProviderAdapter:
    """
    Tries Gemini first. On quota / rate-limit errors (429 / RESOURCE_EXHAUSTED)
    automatically falls back to Groq llama-3.3-70b-versatile.
    """
    def __init__(self, gemini_client, gemini_model, groq_client, groq_model):
        self.gemini_client = gemini_client
        self.gemini_model = gemini_model
        self.groq_client = groq_client
        self.groq_model = groq_model

    def invoke(self, prompt: str) -> SimpleNamespace:
        # 1. Try Gemini models
        if self.gemini_client:
            gemini_models = [
                self.gemini_model,
                "gemini-2.5-flash-lite",
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-2.0-flash",
            ]
            seen_gemini = set()
            ordered_gemini = [m for m in gemini_models if m and not (m in seen_gemini or seen_gemini.add(m))]

            for g_model in ordered_gemini:
                try:
                    response = self.gemini_client.models.generate_content(
                        model=g_model,
                        contents=prompt,
                        config={"temperature": 0.2},
                    )
                    log_llm_provider("Gemini Flash", g_model)
                    return SimpleNamespace(content=getattr(response, "text", ""))
                except Exception as e:
                    error_str = str(e)
                    if _is_quota_error(error_str):
                        log_llm_failover("Gemini Flash", "Groq Llama", "Quota Exceeded (429)")
                    continue

        # 2. Try Groq fast fallback models
        if self.groq_client:
            groq_models = [
                self.groq_model,
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "openai/gpt-oss-120b",
            ]
            # Deduplicate while preserving priority order
            seen_groq = set()
            ordered_groq = [m for m in groq_models if m and not (m in seen_groq or seen_groq.add(m))]

            for q_model in ordered_groq:
                try:
                    completion = self.groq_client.chat.completions.create(
                        model=q_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        max_tokens=1024,
                    )
                    log_llm_provider("Groq Fallback", q_model)
                    text = completion.choices[0].message.content or ""
                    if text:
                        return SimpleNamespace(content=text)
                except Exception:
                    continue

        raise RuntimeError("LLM services temporarily unavailable. Please try again.")


def _is_quota_error(error_str: str) -> bool:
    signals = ["429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "rate_limit"]
    lowered = error_str.lower()
    return any(s in lowered for s in signals)


# ---------------------------------------------------------------------------
# Multilingual NLP Intent & Entity Schema
# ---------------------------------------------------------------------------

class ItemSpec(BaseModel):
    item_name: str
    quantity: int = 1
    unit_hint: Optional[str] = None


class ParsedUserIntent(BaseModel):
    action: Literal["ADD_TO_CART", "REMOVE_ITEM", "VIEW_MENU", "VIEW_CART", "CLEAR_CART", "CHECKOUT", "GENERAL"] = "GENERAL"
    items: List[ItemSpec] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)
    category_preference: Optional[str] = None
    target_reference: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    language_detected: str = "en"
    cleaned_search_query: str = ""
    is_request_for_new_options: bool = False


def parse_intent_with_llm(llm, user_text: str, current_cart_items: Optional[List[dict]] = None) -> ParsedUserIntent:
    """
    Parses user voice or text input using LLM into structured Pydantic schema.
    Dynamically resolves speech homophones, multilingual context, and references using active cart state.
    """
    if not user_text or not user_text.strip():
        return ParsedUserIntent()

    cart_summary = "Empty (0 items)"
    if current_cart_items:
        cart_summary = ", ".join(f"{it.get('quantity', 1)}x {it.get('name', '')}" for it in current_cart_items)

    prompt = f"""
You are an intelligent NLP Parser and Conversational Reasoner for a Voice Command Shopping Assistant.
Analyze the user's voice message with awareness of their current shopping list:

CURRENT CART CONTEXT:
[{cart_summary}]

Return a JSON object matching this schema:
{{
  "action": "ADD_TO_CART" | "REMOVE_ITEM" | "VIEW_MENU" | "VIEW_CART" | "CLEAR_CART" | "CHECKOUT" | "GENERAL",
  "items": [
    {{"item_name": "standard English commodity name", "quantity": 1, "unit_hint": null}}
  ],
  "dietary_preferences": [strings like "gluten_free", "vegan", "vegetarian", "keto_friendly", "organic", "sugar_free", "lactose_free"],
  "category_preference": string or null ("Produce", "Dairy & Eggs", "Bakery", "Pantry & Staples", "Beverages & Snacks"),
  "target_reference": string or null ("first", "second", "third", "both", "it", "that"),
  "max_price": float or null (e.g. 5.0 for "under $5" or "less than 5 dollars"),
  "min_price": float or null,
  "language_detected": "en" | "hi" | "hinglish",
  "cleaned_search_query": "clean English search or removal term",
  "is_request_for_new_options": boolean
}}

Conversational Reasoning & STT Acoustic Correction Rules:
1. "ADD_TO_CART": User wants to add, buy, or restock items.
   - English: "Add milk", "I need 2 apples", "Put bread and eggs on my list", "Running low on butter", "add both", "add it", "add second one".
   - Hindi/Hinglish: "2 packet doodh aur bread add karo", "Mujhe paneer chahiye", "Chai patti aur cheeni khatam ho gayi hai".
   - STT Acoustic Numbers: "add to whole milk" -> quantity: 2, item_name: "whole milk"; "add for apples" -> quantity: 4.
2. "REMOVE_ITEM": User wants to remove, delete, or cancel an item.
   - Contextual Removal: If the user says "remove grief from my card", "Reebok ghee", "delete key", or "remove that", match against CURRENT CART CONTEXT and set "cleaned_search_query" to the exact matching cart item name (e.g. "MorningDew Ghee" or "ghee").
   - Confirmation: If the user says "yes remove", "yes", "remove it", "sure hata do", set action: "REMOVE_ITEM" with target_reference: "it" and "cleaned_search_query" set to the item being confirmed.
3. "CLEAR_CART": User wants to empty or clear list ("Clear my cart", "Empty list", "Sab saaf kar do", "Cart se sab hata do").
4. "VIEW_CART": User asks to see their cart ("Show my list", "What is in my cart?", "Show my updated card", "Mera cart dikhao").
5. "CHECKOUT": User wants to finalize or place order ("Checkout", "Place order", "Order now", "Pay", "place order now").
6. "VIEW_MENU": Browsing or discovering ("Show organic fruits", "Show vegan snacks and green tea").
7. "GENERAL": Greetings or general conversation.

Translation:
- Translate Hindi/Hinglish food items to standard English (doodh -> milk, chai patti -> tea, cheeni -> sugar, tamatar -> tomato, pyaj -> onion, atta -> flour, chawal -> rice, kela -> banana, seb -> apple).

User Message: "{user_text}"

Return ONLY valid JSON with no extra commentary:
"""
    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", "").strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        intent = ParsedUserIntent(**data)
        log_intent_parsed(intent.action, intent.cleaned_search_query or user_text)
        return intent
    except Exception:
        # Fallback heuristic
        lowered = user_text.lower()
        action = "GENERAL"
        if any(w in lowered for w in ["add", "need", "buy", "put", "chahiye", "dalo", "jodo"]):
            action = "ADD_TO_CART"
        elif any(w in lowered for w in ["remove", "delete", "hatao", "cancel"]):
            action = "REMOVE_ITEM"
        elif any(w in lowered for w in ["cart", "list", "dikhao", "show"]):
            action = "VIEW_CART"
        elif any(w in lowered for w in ["checkout", "order", "place"]):
            action = "CHECKOUT"

        intent = ParsedUserIntent(
            action=action,
            cleaned_search_query=user_text,
            items=[ItemSpec(item_name=user_text, quantity=1)] if action == "ADD_TO_CART" else []
        )
        return intent


# ---------------------------------------------------------------------------
# Conversational AI Shopping Concierge
# ---------------------------------------------------------------------------

def get_ai_response(llm, user_input, chat_history, context, memory_context=""):
    """
    Generates a natural, helpful shopping concierge response.
    Never manages raw cart arithmetic — that is handled deterministically in session_memory.py.
    """
    if not llm:
        return "The AI assistant is temporarily unavailable. Please try again in a moment."

    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    prompt = f"""
You are FoodieBot, an intelligent and friendly Voice Shopping Assistant & Supermarket Concierge.
Your goal is to help users manage their grocery shopping lists, discover items, and provide smart recipe & pairing suggestions.

Guidelines:
1. Warm, concise, and helpful tone.
2. If the user spoke in Hindi or Hinglish, reply politely in clear conversational English or friendly Hinglish.
3. Use bullet points (•) to display product recommendations with their price ($X.XX) and unit (e.g. 500g, 1L).
4. Point out relevant dietary badges if applicable (e.g., [Gluten-Free], [Keto-Friendly], [Organic]).
5. After presenting items, suggest smart complementary pairings (e.g. Bread with Butter/Jam; Pasta with Sauce).
6. Do NOT hallucinate prices or invent items not present in the CONTEXT.

{memory_context}

CATALOG CONTEXT:
{context}

CONVERSATION HISTORY (Last 6 turns):
{history_str}

USER MESSAGE:
{user_input}

Respond as FoodieBot:
"""
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "").strip()
        if not content:
            return "Here are some popular supermarket items from our catalog! Let me know which ones you'd like to add."
        return _clean_response(content)
    except Exception as e:
        return f"I encountered a slight technical issue: {e}"


def _clean_response(response: str) -> str:
    """Removes reasoning phrases and formats lines cleanly."""
    if not response:
        return ""
    phrases_to_remove = [
        "Based on your requirements",
        "Based on your request",
        "Looking at our catalog",
        "I would recommend",
    ]
    for p in phrases_to_remove:
        response = response.replace(p, "")
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)
    return response.strip()


from interest_model import predict_intent


def calculate_interest_score(user_input: str, current_score: int, resolved_action=None, search_shown=False) -> int:
    """
    Dual-Layer User Engagement & Intent Scoring:
    Layer 1: ML Model (SentenceTransformer 384-dim Embeddings + Logistic Regression Classifier)
    Layer 2: Deterministic Action Deltas (ADD_TO_CART, REMOVE_ITEM, CLEAR_CART, CHECKOUT)
    """
    score = current_score
    action = resolved_action.get("action", "GENERAL") if isinstance(resolved_action, dict) else "GENERAL"

    # Layer 1: ML Multilingual Sentiment & Tone Prediction
    try:
        sentiment, confidence = predict_intent(user_input)
        if sentiment == "positive":
            score += int(3 + 5 * confidence)
        elif sentiment == "negative":
            score -= int(6 + 8 * confidence)
        elif sentiment == "neutral" and score > 50:
            score -= 2
    except Exception:
        pass

    # Layer 2: Balanced Action Deltas
    if action == "ADD_TO_CART":
        score += 7
    elif action == "CHECKOUT":
        score = max(score + 15, 95)
    elif action == "VIEW_MENU":
        score += 3
    elif action == "REMOVE_ITEM":
        score -= 12
    elif action == "CLEAR_CART":
        score = 30
    elif search_shown and action == "GENERAL":
        score += 1

    final_score = max(5, min(100, score))
    log_interest_score(current_score, final_score, action)
    return final_score