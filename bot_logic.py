# bot_logic.py
import os
import re
from types import SimpleNamespace
import streamlit as st
from google import genai
from groq import Groq
from interest_model import predict_intent

# ---------------------------------------------------------------------------
# LLM Initialization
# ---------------------------------------------------------------------------

@st.cache_resource
def initialize_llm():
    """
    Initializes a dual-provider LLM adapter.
    Primary: Gemini (google-genai)
    Fallback: Groq llama-3.3-70b-versatile  (on 429 / quota errors)
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key   = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
    groq_model = os.getenv("GROQ_MODEL",   "openai/gpt-oss-20b")

    gemini_client = None    
    groq_client   = None

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


from log import log_llm_provider, log_llm_failover, log_intent_parsed, log_interest_score


class _DualProviderAdapter:
    """
    Tries Gemini first. On quota / rate-limit errors (429 / RESOURCE_EXHAUSTED)
    automatically falls back to Groq llama-3.3-70b-versatile.
    """

    def __init__(self, gemini_client, gemini_model, groq_client, groq_model):
        self.gemini_client = gemini_client
        self.gemini_model  = gemini_model
        self.groq_client   = groq_client
        self.groq_model    = groq_model

    def invoke(self, prompt: str) -> SimpleNamespace:
        if self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config={"temperature": 0.2},
                )
                log_llm_provider("Gemini Flash", self.gemini_model)
                return SimpleNamespace(content=getattr(response, "text", ""))
            except Exception as e:
                error_str = str(e)
                if _is_quota_error(error_str):
                    log_llm_failover("Gemini Flash", "Groq Llama 3.3", "Quota Exceeded (429)")
                else:
                    log_llm_failover("Gemini Flash", "Groq Llama 3.3", error_str)

        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                log_llm_provider("Groq Llama 3.3", self.groq_model)
                text = completion.choices[0].message.content or ""
                return SimpleNamespace(content=text)
            except Exception as groq_err:
                raise RuntimeError(f"Both providers failed. Groq error: {groq_err}")

import json
from typing import Literal, List, Optional
from pydantic import BaseModel, Field


class ParsedUserIntent(BaseModel):
    action: Literal["VIEW_MENU", "ADD_TO_CART", "REMOVE_ITEM", "CHECKOUT", "ASK_ALLERGEN", "COMPARE_ITEMS", "GENERAL"] = "GENERAL"
    dietary_restrictions: List[str] = Field(default_factory=list)
    category_preference: Optional[str] = None
    target_reference: Optional[str] = None
    quantity: int = 1
    cleaned_search_query: str = ""
    is_request_for_new_options: bool = False


def parse_intent_with_llm(llm, user_text: str) -> ParsedUserIntent:
    """
    Dynamically parses user input using the LLM into a structured Pydantic schema.
    Handles any typos ("suger", "ingidwnints"), slang, Hindi/Urdu, or complex phrasing
    without a single hardcoded keyword list.
    """
    if not user_text or not user_text.strip():
        return ParsedUserIntent()

    prompt = f"""
You are an expert NLP Intent Parser for a restaurant chatbot called FoodieBot.
Analyze the user's message and return a JSON object matching this schema:

{{
  "action": "VIEW_MENU" | "ADD_TO_CART" | "REMOVE_ITEM" | "CHECKOUT" | "ASK_ALLERGEN" | "COMPARE_ITEMS" | "GENERAL",
  "dietary_restrictions": [strings],
  "category_preference": string or null,
  "target_reference": string or null,
  "quantity": integer (default 1),
  "cleaned_search_query": string,
  "is_request_for_new_options": boolean (true if user asks for alternative, different, or more options/suggestions in any phrasing or language)
}}

Rules:
1. "action":
   - "VIEW_MENU": user wants recommendations, browsing, asking what food exists, cravings, or asking for details/more options.
   - "ADD_TO_CART": user explicitly wants to add/order a specific item or position.
   - "REMOVE_ITEM": user wants to remove an item from cart.
   - "CHECKOUT": user wants to finalize, pay, or complete order.
   - "ASK_ALLERGEN": user asks about allergens, ingredients, or safety.
   - "COMPARE_ITEMS": user asks to compare two items.
   - "GENERAL": greetings, small talk, vague questions.

2. Fix any typos in "cleaned_search_query".
3. Set "is_request_for_new_options" to true whenever the user asks for alternative, different, or more options/suggestions ("any more", "other options", "what else", "change it").

User Message: "{user_text}"

Return ONLY valid JSON with no markdown block or additional text:
"""
    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", "").strip()
        # Clean json backticks if model wrapped in markdown
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("\n", 1)[0].strip()
            if text.startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        intent = ParsedUserIntent(**data)
        log_intent_parsed(intent.action, intent.cleaned_search_query or user_text)
        return intent
    except Exception:
        # Graceful fallback: return default intent
        intent = ParsedUserIntent(cleaned_search_query=user_text)
        log_intent_parsed(intent.action, intent.cleaned_search_query or user_text)
        return intent


def _is_quota_error(error_str: str) -> bool:
    """Returns True when the error is a Gemini quota / rate-limit response."""
    signals = ["429", "RESOURCE_EXHAUSTED", "quota", "rate limit", "rate_limit"]
    lowered = error_str.lower()
    return any(s.lower() in lowered for s in signals)


# ---------------------------------------------------------------------------
# Response Generation
# ---------------------------------------------------------------------------

def get_ai_response(llm, user_input, chat_history, context, memory_context=""):
    """
    Generates a conversational FoodieBot response using the LLM.
    Cart mutations and order confirmations never reach this function's
    output — those are handled entirely in session_memory.py / ui_components.py.
    """
    if not llm:
        return "The language model is not available. Please try again later."

    if _is_inappropriate_or_irrelevant(user_input):
        return "My apologies, but I can only provide information about our menu items."

    history_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history]
    )

    prompt = f"""
You are FoodieBot, a warm, polite, and knowledgeable gourmet restaurant concierge. Your tone is welcoming, hospitable, concise, and helpful. Follow these rules:

- Gracefully handle informal language, minor typos, slang, and multilingual queries by providing clear, hospitable menu guidance.
- When asked to pick or compare specific options from previous suggestions, directly name the single top matching item and briefly explain why, instead of re-listing multiple items.
- When asked to describe or elaborate on a specific item (e.g., "tell about this item", "describe this dish"), write a warm 2-3 sentence conversational overview explaining its flavor profile, key ingredients, and appeal, rather than just re-listing raw bullet points.
- NEVER use internal reasoning phrases like "Based on your request" or "I would recommend".
- NEVER output any internal scores, numbers, or system metrics in your response.
- Use bullet points (•) to list items. Show prices as $X.XX, include calories, category, allergens if known.
- When listing items, use ONE bullet for the item name and price, then indent the details (calories, category, allergens) on the following lines with 4 spaces.
  Example:
  • Thai Peanut Crunch Salad – $8.99
      \tCalories: 450
      \tCategory: Salads & Healthy Options
      \tAllergens: Contains: peanuts; soy
- You do NOT manage the cart. Never say an item was "added", "removed", or that the order was "updated" — cart changes are handled separately and shown to the user directly. If asked what's in the cart, you may reference the SESSION MEMORY `Ordered items` list read-only, but do not claim to modify it.
- Use the CONTEXT to answer questions. If the user asks if an item is healthy, light, or heavy, use its calorie count and ingredients to respond (e.g., "It has 610 calories and is fried, so it's an indulgent option"). If the user compares items, you may do so.
- If the user's request is contradictory, missing critical information, or unclear, politely ask ONE clarifying question before attempting to answer.
- When the user asks for a holistic recommendation based on their preferences, carefully review the SESSION MEMORY and the conversation to identify their stated dietary restrictions, likes, and dislikes, and suggest the single best matching item from the menu.
- If the user mentions non-food, chemical, or hazardous household substances (e.g., detergent, Harpic, bleach, soap, phenyl, washing powder), politely and firmly explain that chemical cleaning products are toxic and strictly unsafe for consumption, and offer to help them find a safe, delicious menu item instead.
- If exact parameters (like sugar grams or specific recipe steps) are not in the CONTEXT, answer conversationally using available menu items and categories (e.g., recommend Salads, Unsweetened Beverages, or light options) instead of giving abrupt non-answers.

{memory_context}

CONTEXT:
{context}

CONVERSATION HISTORY:
{history_str}

USER MESSAGE:
{user_input}

Respond as FoodieBot.
"""
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", "")
        if not content or not content.strip():
            if context and "No relevant items" not in context:
                return f"Here are a few more delicious options from our menu:\n\n{context}"
            return "Here are a few more popular options! Let me know if any of these sound good to you."
        return _clean_response(content)
    except Exception as e:
        return f"Sorry, I'm having a technical issue right now: {e}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_inappropriate_or_irrelevant(user_input):
    """
    Delegates moderation and off-topic handling to the LLM system prompt rules.
    Only flags non-text/corrupted data inputs.
    """
    if len(user_input) > 10:
        non_ascii_count = sum(1 for c in user_input if ord(c) > 127)
        if non_ascii_count > len(user_input) * 0.7:
            return True

    return False


def _clean_response(response):
    """Removes reasoning phrases AND any leaked internal scores/metrics."""
    if not response:
        return ""
    phrases_to_remove = [
        "Based on your requirements",
        "Based on your request",
        "I would recommend",
        "I'll exclude",
        "Looking at the menu",
    ]
    for phrase in phrases_to_remove:
        response = response.replace(phrase, "")

    response = re.sub(r'(?i)\b(interest\s*)?score\s*:?\s*\d+\b', '', response)
    response = re.sub(r'\n\s*\n\s*\n', '\n\n', response)

    # Ensure single newlines in item detail lines (Calories, Category, Allergens)
    # end with two spaces so Streamlit Markdown does not collapse them into one line.
    lines = response.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.rstrip()
        if any(keyword in stripped for keyword in ["Calories:", "Category:", "Allergens:", "•"]):
            cleaned_lines.append(stripped + "  ")
        else:
            cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


# ---------------------------------------------------------------------------
# Interest scoring — phrase-based (tone) + action-based (resolved outcome)
# ---------------------------------------------------------------------------

ACTION_SCORE_DELTAS = {
    "VIEW_MENU": 3,
    "ASK_ALLERGEN": 4,
    "COMPARE_ITEMS": 5,
    "ADD_TO_CART": 15,
    "REMOVE_ITEM": -8,
    "CHECKOUT": 20,
    "GENERAL": 0,
}

# Negation guard: any of these appearing before/around a "positive" phrase
# flips it. Word-boundary regex, not naive substring, so "didnt" without an
# apostrophe still matches (unlike the original " not "/"don't" list, which
# missed "didnt add it" entirely and let "add it" score +10 as if it were
# a real order).
_NEGATION_PATTERN = re.compile(
    r"\b(not|dont|don't|didnt|didn't|never|no|cancel|remove|without|isn't|isnt|wasn't|wasnt)\b"
)


def _has_negation(text):
    return bool(_NEGATION_PATTERN.search(text))


def _phrase_based_score(user_input, score):
    """Tone/sentiment layer.

    Only two explicit override tiers remain — unambiguous hard negations that
    the ML model might still score as mildly positive. Everything else
    (affirmatives, mild positives, neutral queries) is delegated to
    predict_intent so new phrasings are handled automatically.
    """
    lowered = user_input.lower()

    # Greetings / Small-talk should never penalize initial score below baseline 50
    greeting_patterns = [r"\b(hello|hi|hey|hanji|hnji|namaste|good morning|good evening)\b", r"\bwhat'?s happening\b"]
    if any(re.search(p, lowered) for p in greeting_patterns) and score <= 50:
        return 50

    # These phrases unambiguously cancel or refuse an action. Keep them as
    # hard overrides so a fragile model confidence can't flip them.
    negative_action_phrases = [
        "do not add", "don't add", "dont add", "didnt add", "didn't add",
        "do not order", "don't order", "dont order", "cancel that",
        "cancel my order", "remove that",
    ]
    rejection_phrases = [
        "no thanks", "not interested", "don't want", "dont want", "not now",
        "maybe later", "leave it", "different item",
    ]

    if any(phrase in lowered for phrase in negative_action_phrases):
        return score - 18

    if any(phrase in lowered for phrase in rejection_phrases):
        return score - 10

    has_negation = _has_negation(lowered)
    # A contrast word reverses the scope of the negation:
    # "I don't like spicy but add it anyway" is still a genuine order.
    has_contrast = bool(re.search(r'\bbut\b|\bhowever\b|\banyway\b|\bstill\b', lowered))

    if has_negation and not has_contrast:
        return score - 8

    # Delegate everything else — positive orders, affirmatives, neutral queries
    # — to the trained intent model.
    try:
        intent, confidence = predict_intent(user_input)
        if intent == "positive" and confidence >= 0.5:
            return score + int(2 + 6 * confidence)
        elif intent == "negative":
            return score - int(4 + 8 * confidence)
        elif intent == "neutral" and score > 50:
            return score - 1
    except Exception:
        pass
    return score


def _action_based_score(resolved_action, score):
    """Scores the RESOLVED outcome from
    session_memory.update_state_from_user_message()."""
    if not resolved_action:
        return score

    action = resolved_action.get("action", "GENERAL")

    if action in ("ADD_TO_CART", "REMOVE_ITEM", "CHECKOUT") and not resolved_action.get("cart_changed"):
        return score

    delta = ACTION_SCORE_DELTAS.get(action, 0)
    return score + delta


def calculate_interest_score(user_input, current_score, resolved_action=None, search_shown=False):
    """
    Combined score: phrase-based tone signal + action-based outcome signal.
    Pass `resolved_action` (the dict from update_state_from_user_message())
    and `search_shown` (True when a relevant search result was returned)
    through from ui_components.py.
    """
    score = current_score
    score = _phrase_based_score(user_input, score)
    score = _action_based_score(resolved_action, score)
    # If relevant items were surfaced but the action stayed GENERAL (e.g.
    # "i think some spicy" doesn't match any VIEW_MENU keyword), add a
    # small nudge so genuine menu exploration is reflected in the score.
    if search_shown and resolved_action and resolved_action.get("action") == "GENERAL":
        score += 3
    final_score = max(0, min(100, score))
    action_name = resolved_action.get("action", "GENERAL") if isinstance(resolved_action, dict) else "GENERAL"
    log_interest_score(current_score, final_score, action_name)
    return final_score