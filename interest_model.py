import os
import logging
import warnings
from functools import lru_cache

# Silence HuggingFace weight loading and tokenizer progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import streamlit as st
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


TRAINING_DATA = [
    # ── Positive: Explicit Grocery Shopping Actions ─────────────────────────
    ("add it to my shopping list", "positive"),
    ("add to my list", "positive"),
    ("add it", "positive"),
    ("add this", "positive"),
    ("add that", "positive"),
    ("add both", "positive"),
    ("yes add it", "positive"),
    ("sure add that", "positive"),
    ("add 2 milk", "positive"),
    ("add bread and eggs", "positive"),
    ("put apples on my list", "positive"),
    ("i want to buy this", "positive"),
    ("i want to order now", "positive"),
    ("i need milk", "positive"),
    ("i am out of coffee", "positive"),
    ("we need more groceries", "positive"),
    ("put it in my cart", "positive"),
    ("add to cart", "positive"),
    ("buy this item", "positive"),
    ("get me this", "positive"),
    ("place the order", "positive"),
    ("ready to checkout", "positive"),
    ("checkout now", "positive"),

    # ── Positive: Affirmatives & Approvals ───────────────────────────────────
    ("yes please", "positive"),
    ("yes", "positive"),
    ("ok", "positive"),
    ("okay", "positive"),
    ("sure", "positive"),
    ("alright", "positive"),
    ("yep", "positive"),
    ("yeah", "positive"),
    ("go ahead", "positive"),
    ("sounds good", "positive"),
    ("that looks great", "positive"),
    ("perfect choice", "positive"),
    ("looks fresh", "positive"),
    ("i like this brand", "positive"),

    # ── Positive: Restock & Craving Signals ──────────────────────────────────
    ("running low on groceries", "positive"),
    ("we need fresh fruits", "positive"),
    ("need vegetables for dinner", "positive"),
    ("need essentials for home", "positive"),

    # ── Positive: Contrast & Preference Statements ───────────────────────────
    ("not white bread but add whole wheat", "positive"),
    ("i like this but without dairy", "positive"),
    ("i want organic milk", "positive"),
    ("don't have sugar at home add it anyway", "positive"),

    # ── Negative: Refusals, Cancellations, Removals ──────────────────────────
    ("no thanks", "negative"),
    ("not interested", "negative"),
    ("i don't need this", "negative"),
    ("we already have milk at home", "negative"),
    ("already have that", "negative"),
    ("remove that", "negative"),
    ("remove bread from list", "negative"),
    ("cancel that", "negative"),
    ("clear my cart", "negative"),
    ("empty the list", "negative"),
    ("do not add it", "negative"),
    ("don't add that", "negative"),
    ("leave it", "negative"),
    ("not now", "negative"),
    ("maybe next time", "negative"),
    ("nah", "negative"),
    ("skip it", "negative"),
    ("too expensive", "negative"),

    # ── Neutral: Grocery Browsing, Prices, & Dietary Inquiries ──────────────
    ("show me options", "neutral"),
    ("what fresh fruits do you have", "neutral"),
    ("show me dairy items", "neutral"),
    ("what snacks are available", "neutral"),
    ("what do you recommend", "neutral"),
    ("tell me more about this product", "neutral"),
    ("what is the unit size", "neutral"),
    ("what is the price", "neutral"),
    ("show cheaper options", "neutral"),
    ("is this organic", "neutral"),
    ("is it gluten free", "neutral"),
    ("is this keto friendly", "neutral"),
    ("is it lactose free", "neutral"),
    ("does it have added sugar", "neutral"),
    ("what are the ingredients", "neutral"),
    ("compare these two items", "neutral"),
    ("which brand is better", "neutral"),

    # ── Positive: Hindi / Hinglish Grocery Order & Restock Signals ───────────
    ("haan add kar do", "positive"),
    ("list me daal do", "positive"),
    ("doodh add karo", "positive"),
    ("paneer chahiye mujhe", "positive"),
    ("chawal aur tel le lo", "positive"),
    ("ghar pe doodh khatam ho gaya", "positive"),
    ("ration order karna hai", "positive"),
    ("haan dono add kar do", "positive"),
    ("ye wala le lo", "positive"),

    # ── Negative: Hindi / Hinglish Refusals & Cancellations ──────────────────
    ("nhi chahiye", "negative"),
    ("hata do list se", "negative"),
    ("cancel kar do", "negative"),
    ("ye mat lena", "negative"),
    ("ghar pe pehle se hai", "negative"),
    ("bohot mehenga hai", "negative"),

    # ── Neutral: Hindi / Hinglish Grocery Inquiries ──────────────────────────
    ("kya options hain", "neutral"),
    ("price kitna hai", "neutral"),
    ("konse fruits available hain", "neutral"),
    ("organic items dikhao", "neutral"),
    ("gluten free snacks hain kya", "neutral"),
]


LABEL_TO_INDEX = {"negative": 0, "neutral": 1, "positive": 2}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}

_CACHED_ENCODER = None
_CACHED_MODEL = None


def get_cached_encoder():
    """Singleton cached encoder in process memory."""
    global _CACHED_ENCODER
    if _CACHED_ENCODER is None:
        _CACHED_ENCODER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _CACHED_ENCODER


def get_intent_model():
    """Trains and returns the cached Multinomial Logistic Regression model."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        encoder = get_cached_encoder()
        texts = [text for text, _ in TRAINING_DATA]
        labels = [LABEL_TO_INDEX[label] for _, label in TRAINING_DATA]

        X = encoder.encode(texts, show_progress_bar=False)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X, labels)
        _CACHED_MODEL = clf
    return _CACHED_MODEL


def predict_intent(user_input: str):
    """
    Predicts user purchase intent and returns (label, confidence).
    Labels: 'positive' (buying intent), 'neutral' (browsing/asking), 'negative' (refusals/removals)
    """
    encoder = get_cached_encoder()
    clf = get_intent_model()
    vec = encoder.encode([user_input], show_progress_bar=False)
    probabilities = clf.predict_proba(vec)[0]
    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    return INDEX_TO_LABEL[predicted_index], confidence