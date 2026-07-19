# interest_model.py
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


TRAINING_DATA = [
    ("add it to my order", "positive"),
    ("i want this", "positive"),
    ("i want the burger", "positive"),
    ("i want a meal", "positive"),
    ("place the order", "positive"),
    ("yes please", "positive"),
    ("yes add it", "positive"),
    ("okay add the burger", "positive"),
    ("sure add that", "positive"),
    ("i'll take it", "positive"),
    ("i will take this", "positive"),
    ("that looks great", "positive"),
    ("sounds good", "positive"),
    ("i love this", "positive"),
    ("perfect choice", "positive"),
    ("i am hungry", "positive"),
    ("i am starving", "positive"),
    ("i want to order now", "positive"),
    ("no thanks", "negative"),
    ("not interested", "negative"),
    ("i am not interested", "negative"),
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
    ("not drinks but i want burger", "positive"),
    ("i like this but no dairy", "positive"),
    ("i want something but not spicy", "positive"),
]


LABEL_TO_INDEX = {"negative": 0, "neutral": 1, "positive": 2}
INDEX_TO_LABEL = {index: label for label, index in LABEL_TO_INDEX.items()}


@lru_cache(maxsize=1)
def get_intent_model():
    texts = [text for text, _ in TRAINING_DATA]
    labels = [LABEL_TO_INDEX[label] for _, label in TRAINING_DATA]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 3), lowercase=True),
            ),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    model.fit(texts, labels)    
    return model


def predict_intent(user_input):
    model = get_intent_model()
    probabilities = model.predict_proba([user_input])[0]
    predicted_index = int(probabilities.argmax())
    confidence = float(probabilities[predicted_index])
    return INDEX_TO_LABEL[predicted_index], confidence