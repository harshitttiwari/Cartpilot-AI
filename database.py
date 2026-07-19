# database.py
import re
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from data_pipeline import load_and_process_menu_data

DATA_FILE_PATH = "fast_food_products.csv"

@st.cache_resource
def initialize_services():
    """Load CSV, initialize ChromaDB, and build embeddings."""
    try:
        df, analysis = load_and_process_menu_data(DATA_FILE_PATH)
    except FileNotFoundError:
        st.error("CSV file not found. Please place 'fast_food_products.csv' in the same folder as app.py.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None

    try:
        client = chromadb.EphemeralClient()
        try:
            client.delete_collection("food_items")
        except Exception:
            pass
        collection = client.create_collection(
            "food_items",
            metadata={"hnsw:space": "cosine"},
        )
    except Exception as e:
        st.error(f"Failed to initialize ChromaDB: {e}")
        return None, None, None, None, None, None

    try:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Failed to load embedding model: {e}")
        return None, None, None, None, None, None

    try:
        records = df.to_dict("records")
        metadatas = [{str(k): str(v) for k, v in rec.items()} for rec in records]
        ids = [str(m["product_id"]) for m in metadatas]

        documents = [
            f"Item Name: {r.get('name', '')}. "
            f"Description: {r.get('description', '')}. "
            f"Ingredients: {r.get('ingredients', '')}. "
            f"Calories: {r.get('calories', 'N/A')}. "
            f"Allergens: {r.get('allergens', 'None listed')}. "
            f"Dietary Tags: {r.get('dietary_tags', '')}."
            for r in metadatas
        ]
        embeddings = embedder.encode(documents).tolist()
        collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

        tokenized_documents = [
            _tokenize_text(row.get("search_text") or documents[index])
            for index, row in enumerate(records)
        ]
        bm25 = BM25Okapi(tokenized_documents)
    except Exception as e:
        st.error(f"Failed to build embeddings: {e}")
        return None, None, None, None, None, None

    return df, collection, embedder, analysis, bm25, records


def _tokenize_text(text):
    """Lowercase, strip punctuation, split — so 'burger,' and 'burger' are the same token."""
    cleaned = re.sub(r"[^\w\s]", " ", str(text).lower())
    return [token for token in cleaned.split() if token]