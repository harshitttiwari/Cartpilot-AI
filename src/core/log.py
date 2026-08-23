# log.py
"""
Voice Command Shopping Assistant Centralized System & Model Logging Engine
Provides clean, color-coded terminal logs for:
- Voice Command Captures & Multilingual Detection (English, Hindi)
- LLM Provider Selection (Gemini Flash vs Groq Llama 3.3)
- Embedding Generation & Vector Search (ChromaDB + BM25)
- Pydantic Intent Parsing & Entity Extraction
- Shopping Cart State & Smart Suggestions
- User Engagement Score Progression
"""

import logging
import sys
import os

# Color codes for terminal logging
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"


class ShoppingAssistantFormatter(logging.Formatter):
    """Custom logging formatter with timestamp and color coding."""
    def __init__(self, use_color=True):
        super().__init__(
            fmt="%(asctime)s | [%(levelname)s] | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.use_color = use_color

    def format(self, record):
        if not self.use_color:
            # Strip ANSI color codes for file logging
            msg = super().format(record)
            import re
            return re.sub(r'\033\[[0-9;]*m', '', msg)
        return super().format(record)


# Initialize logger
logger = logging.getLogger("VoiceShoppingAssistant")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    # 1. Console StreamHandler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ShoppingAssistantFormatter(use_color=True))
    logger.addHandler(console_handler)

    # 2. Persistent FileHandler (foodiebot.log)
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foodiebot.log")
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(ShoppingAssistantFormatter(use_color=False))
    logger.addHandler(file_handler)


def log_voice_command(language: str, transcript: str):
    """Logs voice recognition transcript and detected language."""
    msg = f"{COLOR_CYAN}[VOICE INPUT]{COLOR_RESET} Lang: {COLOR_YELLOW}{language}{COLOR_RESET} | Transcript: '{transcript}'"
    logger.info(msg)


def log_llm_provider(provider_name: str, model_name: str):
    """Logs active LLM model selection."""
    msg = f"{COLOR_CYAN}[LLM ADAPTER]{COLOR_RESET} Selected Provider: {COLOR_GREEN}{provider_name}{COLOR_RESET} ({model_name})"
    logger.info(msg)


def log_llm_failover(primary: str, fallback: str, reason: str):
    """Logs automatic failover switch between primary and fallback LLM."""
    msg = f"{COLOR_YELLOW}[LLM FAILOVER]{COLOR_RESET} Primary {primary} failed ({reason}) -> Switched to Fallback: {COLOR_GREEN}{fallback}{COLOR_RESET}"
    logger.warning(msg)


def log_embedding_generated(query_text: str, dim: int = 384):
    """Logs 384-dimensional transformer embedding generation."""
    msg = f"{COLOR_BLUE}[EMBEDDING ENGINE]{COLOR_RESET} Generated {dim}-dim vector for query: '{query_text[:40]}...'"
    logger.info(msg)


def log_vector_search(query: str, results_count: int, top_relevance: float):
    """Logs ChromaDB + BM25 hybrid search execution."""
    msg = f"{COLOR_MAGENTA}[HYBRID SEARCH]{COLOR_RESET} Query: '{query}' | Results: {results_count} items | Top Relevance: {top_relevance:.2%}"
    logger.info(msg)


def log_intent_parsed(action: str, cleaned_query: str):
    """Logs Pydantic structured intent extraction."""
    msg = f"{COLOR_GREEN}[INTENT PARSER]{COLOR_RESET} Parsed Action: {COLOR_YELLOW}{action}{COLOR_RESET} | Cleaned Query: '{cleaned_query}'"
    logger.info(msg)


def log_cart_action(action: str, item_name: str, quantity: int, cart_total: float):
    """Logs shopping cart mutations and calculated totals."""
    msg = f"{COLOR_GREEN}[CART ENGINE]{COLOR_RESET} Action: {action} | Item: {quantity}x {item_name} | New Total: ${cart_total:.2f}"
    logger.info(msg)


def log_interest_score(old_score: int, new_score: int, turn_action: str):
    """Logs user interest score trajectory update."""
    msg = f"{COLOR_CYAN}[ENGAGEMENT SCORE]{COLOR_RESET} Trajectory: {old_score}% -> {COLOR_GREEN}{new_score}%{COLOR_RESET} (Action: {turn_action})"
    logger.info(msg)
