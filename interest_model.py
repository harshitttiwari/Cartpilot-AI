# interest_model.py
from functools import lru_cache

import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


TRAINING_DATA = [
    # ── Positive: explicit order actions ──────────────────────────────────
    ("add it to my order", "positive"),
    ("add it", "positive"),
    ("add this", "positive"),
    ("add that", "positive"),
    ("order it", "positive"),
    ("order this", "positive"),
    ("place the order", "positive"),
    ("yes add it", "positive"),
    ("okay add the burger", "positive"),
    ("sure add that", "positive"),
    ("i'll take it", "positive"),
    ("i will take this", "positive"),
    ("i want this", "positive"),
    ("i want the burger", "positive"),
    ("i want a meal", "positive"),
    ("i want to order now", "positive"),
    ("give me that", "positive"),
    ("get me this", "positive"),
    # ── Positive: affirmatives ────────────────────────────────────────────
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
    ("i love this", "positive"),
    ("perfect choice", "positive"),
    ("looks perfect", "positive"),
    # ── Positive: hunger / craving signals ───────────────────────────────
    ("i am hungry", "positive"),
    ("i am starving", "positive"),
    ("i am craving something spicy", "positive"),
    ("feeling hungry", "positive"),
    # ── Positive: contrast-negation (negation applies to a different clause)
    ("not drinks but i want burger", "positive"),
    ("i like this but no dairy", "positive"),
    ("i want something but not spicy", "positive"),
    ("i don't like spicy but add it anyway", "positive"),
    # ── Negative: refusals and cancellations ─────────────────────────────
    ("no thanks", "negative"),
    ("not interested", "negative"),
    ("i am not interested", "negative"),
    ("i am full no thanks", "negative"),
    ("full no thanks", "negative"),
    ("different item", "negative"),
    ("show another item", "negative"),
    ("cancel that", "negative"),
    ("cancel my order", "negative"),
    ("remove that", "negative"),
    ("do not add it", "negative"),
    ("don't add it", "negative"),
    ("do not order this", "negative"),
    ("exit", "negative"),
    ("leave it", "negative"),
    ("maybe later", "negative"),
    ("not now", "negative"),
    ("do not want this", "negative"),
    ("nah", "negative"),
    ("skip it", "negative"),
    # ── Neutral: browsing and detail queries ─────────────────────────────
    ("show me options", "neutral"),
    ("show me another option", "neutral"),
    ("what do you recommend", "neutral"),
    ("tell me more", "neutral"),
    ("how spicy is it", "neutral"),
    ("what are the ingredients", "neutral"),
    ("is it vegetarian", "neutral"),
    ("is it vegan", "neutral"),
    ("is it gluten free", "neutral"),
    ("does it have dairy", "neutral"),
    ("compare this with another item", "neutral"),
    ("can you explain the calories", "neutral"),
    ("show cheaper options", "neutral"),
    ("what is the price", "neutral"),
    ("rate this dish out of 10", "neutral"),
    ("how hot is this on a scale", "neutral"),
    # ── Positive: Hindi / Hinglish hunger & order signals ──────────────────
    ("bhuk lag rha hai", "positive"),
    ("bhuk lag rha merko", "positive"),
    ("bohot bhuk lagi hai", "positive"),
    ("haan add kar do", "positive"),
    ("spicy khana chahiye", "positive"),
    ("kya menu me spicy pizza hai", "positive"),
    # ── Negative: Hindi / Hinglish refusals ───────────────────────────────
    ("nhi chahiye", "negative"),
    ("bekaar hai", "negative"),
    ("bohot bekaar food", "negative"),
    ("cancel kar do", "negative"),
    # ── Neutral: Hindi / Hinglish queries ──────────────────────────────────
    ("kya options hain", "neutral"),
    ("price kitna hai", "neutral"),
]


LABEL_TO_INDEX = {"negative": 0, "neutral": 1, "positive": 2}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


@st.cache_resource
def _load_encoder():
    """Load the multilingual sentence encoder once per process.

    paraphrase-multilingual-MiniLM-L12-v2 handles English, Hindi, Urdu,
    and 50+ other languages — so mixed-language inputs like "bhuk lag rha"
    or "haan chalega" are understood without adding hardcoded phrase lists.
    """
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


@lru_cache(maxsize=1)
def get_intent_model():
    encoder = _load_encoder()
    texts = [text for text, _ in TRAINING_DATA]
    labels = [LABEL_TO_INDEX[label] for _, label in TRAINING_DATA]

    X = encoder.encode(texts)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X, labels)
    return clf


def predict_intent(user_input):
    encoder = _load_encoder()
    clf = get_intent_model()
    vec = encoder.encode([user_input])
    probabilities = clf.predict_proba(vec)[0]
    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    return INDEX_TO_LABEL[predicted_index], confidence