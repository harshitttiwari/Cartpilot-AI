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
                return SimpleNamespace(content=getattr(response, "text", ""))
            except Exception as e:
                error_str = str(e)
                if _is_quota_error(error_str):
                    pass
                else:
                    pass

        if self.groq_client:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=1024,
                )
                text = completion.choices[0].message.content or ""
                return SimpleNamespace(content=text)
            except Exception as groq_err:
                raise RuntimeError(f"Both providers failed. Groq error: {groq_err}")

        raise RuntimeError("No LLM provider available.")


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
You are FoodieBot, a professional restaurant assistant. Follow these rules:

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
- Only say "I don't have that information" if the CONTEXT truly lacks the required data.

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
            return "I'm sorry, I didn't quite catch that. Could you please rephrase your request?"
        return _clean_response(content)
    except Exception as e:
        return f"Sorry, I'm having a technical issue and can't respond right now. Error: {e}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_inappropriate_or_irrelevant(user_input):
    """Checks for inappropriate content or off-topic messages."""
    bad_words = ['sex', 'porn', 'fuck', 'shit']
    if any(word in user_input.lower() for word in bad_words):
        return True

    if len(user_input) > 10:
        non_ascii_count = sum(1 for c in user_input if ord(c) > 127)
        if non_ascii_count > len(user_input) * 0.7:
            return True

    off_topic = ['weather', 'politics']
    if any(word in user_input.lower() for word in off_topic):
        food_words = ['food', 'eat', 'menu', 'hungry', 'burger', 'pizza', 'order']
        if not any(word in user_input.lower() for word in food_words):
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
    return response.strip()


# ---------------------------------------------------------------------------
# Interest scoring — phrase-based (tone) + action-based (resolved outcome)
# ---------------------------------------------------------------------------

ACTION_SCORE_DELTAS = {
    "VIEW_MENU": 5,
    "ASK_ALLERGEN": 8,
    "COMPARE_ITEMS": 10,
    "ADD_TO_CART": 30,
    "REMOVE_ITEM": -10,
    "CHECKOUT": 40,
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
    """Tone/sentiment layer. Kept intentionally small — do not keep adding
    more phrasings here; that's what the action-based layer is for."""
    lowered = user_input.lower()

    negative_action_phrases = [
        "do not add", "don't add", "dont add", "didnt add", "didn't add",
        "do not order", "don't order", "dont order", "cancel that",
        "cancel my order", "remove that",
    ]
    rejection_phrases = [
        "no thanks", "not interested", "don't want", "dont want", "not now",
        "maybe later", "leave it", "different item",
    ]
    strong_order_phrases = [
        "add it", "add this", "add that", "place order", "order it",
        "order this", "i'll take", "i will take", "checkout",
    ]
    positive_phrases = [
        "yes", "sure", "okay", "ok", "sounds good", "looks good",
        "looks great", "love", "perfect", "want",
    ]

    if any(phrase in lowered for phrase in negative_action_phrases):
        return score - 18

    if any(phrase in lowered for phrase in rejection_phrases):
        return score - 10

    has_negation = _has_negation(lowered)

    # Negation guard: a "positive" phrase inside a negated sentence
    # ("didnt add it", "you didn't add it") must NOT score as positive.
    matches_strong_order = any(phrase in lowered for phrase in strong_order_phrases)
    matches_positive = any(phrase in lowered for phrase in positive_phrases)

    if has_negation and (matches_strong_order or matches_positive):
        # Negated positive phrase — treat as mild negative, not a reward.
        return score - 6

    if has_negation:
        return score - 8

    if matches_strong_order:
        return score + 10

    if matches_positive:
        return score + 4

    try:
        intent, confidence = predict_intent(user_input)
        if intent == "positive" and confidence >= 0.5:
            return score + int(2 + 6 * confidence)
        elif intent == "negative":
            return score - int(4 + 8 * confidence)
        elif intent == "neutral" and score > 50:
            return score - 1
    except Exception:
        keywords_boost = {
            10: ["add it", "order it", "i'll take", "yes add", "place order"],
            4:  ["want", "get", "take", "perfect", "love", "great", "good"],
            3:  ["hungry", "starving", "craving"],
            -10: ["no thanks", "not interested", "don't want", "different", "exit"],
        }
        for boost, words in keywords_boost.items():
            if any(word in lowered for word in words):
                return score + boost
                break

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


def calculate_interest_score(user_input, current_score, resolved_action=None):
    """
    Combined score: phrase-based tone signal + action-based outcome signal.
    Pass `resolved_action` (the dict from update_state_from_user_message())
    through from ui_components.py.
    """
    score = current_score
    score = _phrase_based_score(user_input, score)
    score = _action_based_score(resolved_action, score)
    return max(0, min(100, score))