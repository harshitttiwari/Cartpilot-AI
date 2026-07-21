# log.py
"""
FoodieBot Centralized System & Model Logging Engine
Provides clean, color-coded terminal logs for:
- LLM Provider Selection (Gemini Flash vs Groq Llama 3.3)
- Embedding Generation & Vector Search (ChromaDB + BM25)
- Pydantic Intent Parsing
- Interest Score Progression
"""

import logging
import sys

# Color codes for terminal logging
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"


class FoodieBotFormatter(logging.Formatter):
    """Custom logging formatter with timestamp and color coding."""
    FORMAT = "%(asctime)s | [%(levelname)s] | %(message)s"

    def format(self, record):
        log_fmt = self.FORMAT
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


# Initialize logger
logger = logging.getLogger("FoodieBot")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FoodieBotFormatter())
    logger.addHandler(handler)


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


def log_interest_score(old_score: int, new_score: int, turn_action: str):
    """Logs user interest score trajectory update."""
    msg = f"{COLOR_CYAN}[INTEREST SCORE]{COLOR_RESET} Trajectory: {old_score}% -> {COLOR_GREEN}{new_score}%{COLOR_RESET} (Action: {turn_action})"
    logger.info(msg)
