# 🤖 FoodieBot - Enterprise AI Restaurant Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-purple?style=for-the-badge)](https://www.trychroma.com/)
[![Gemini & Groq](https://img.shields.io/badge/LLM-Gemini_2.5_Flash_%2B_Groq_Llama_3.3-orange?style=for-the-badge)](https://ai.google.dev/)

An enterprise-grade, 100% Pure Python full-stack AI restaurant assistant. Built with **FastAPI**, **Streamlit**, **ChromaDB Hybrid Search**, **Dual LLM Provider Failover (Gemini 2.5 Flash + Groq Llama 3.3)**, and a **Multilingual Sentiment & Interest Engine**.

---

## 🌟 Architecture & Key Innovations

```
                               ┌────────────────────────────────────────┐
                               │       Streamlit Web Chatbot UI         │
                               │        (Frontend - Port 8501)          │
                               └──────────────────┬─────────────────────┘
                                                  │ HTTP POST /api/chat
                                                  ▼
 ┌──────────────────────────────────────────────────────────────────────────────────────────────┐
 │                              FastAPI REST Backend (Port 8000)                                │
 ├────────────────────────────┬───────────────────────────────┬─────────────────────────────────┤
 │     1. Dual LLM Failover   │   2. Hybrid Search Engine     │ 3. Memory & State Anchor Engine │
 │  ┌──────────────────────┐  │  ┌─────────────────────────┐  │  ┌───────────────────────────┐  │
 │  │ Gemini 2.5 Flash     │  │  │ ChromaDB HNSW Vector    │  │  │ Pydantic Intent Parsing   │  │
 │  │ (Primary)            │  │  │ (384-dim Embeddings)   │  │  │ (Zero Hardcoded Rules)    │  │
 │  └──────────┬───────────┘  │  └────────────┬────────────┘  │  └─────────────┬─────────────┘  │
 │             │ 429 Quota    │               │               │                │                │
 │             ▼              │               ▼               │                ▼                │
 │  ┌──────────────────────┐  │  ┌─────────────────────────┐  │  ┌───────────────────────────┐  │
 │  │ Groq Llama 3.3 70B   │  │  │ Okapi BM25 Keyword      │  │  │ Pronoun ("add it") &      │  │
 │  │ (Automatic Fallback) │  │  │ Ranker                  │  │  │ Ordinal ("1st") Anchoring │  │
 │  └──────────────────────┘  │  └─────────────────────────┘  │  └───────────────────────────┘  │
 └────────────────────────────┴───────────────────────────────┴─────────────────────────────────┘
                                                  │
                                                  ▼
                               ┌────────────────────────────────────┐
                               │   Centralized Logger (log.py)      │
                               │   (Real-Time Color Terminal Logs)  │
                               └────────────────────────────────────┘
```

### 1. 🛡️ Dual-Provider LLM Failover (`bot_logic.py`)
- **Primary**: Google **Gemini 2.5 Flash** (`gemini-2.5-flash`) for lightning-fast gourmet recommendations.
- **Automatic Failover**: On encountering `429 Rate Limit / Quota Exceeded` errors, the system automatically redirects requests to **Groq Llama 3.3 70B** (`llama-3.3-70b-versatile`) with zero user interruption.

### 2. ⚡ FastAPI REST Server (`api.py`)
- Exposes clean, high-performance REST API endpoints (`/health`, `/api/chat`, `/api/menu`).
- Interactive OpenAPI / Swagger Documentation available live at `http://localhost:8000/docs`.

### 3. 🔍 Hybrid Search Engine (`ui_components.py` & `database.py`)
- Combines dense vector retrieval via **ChromaDB HNSW ANN Index** (384-dimensional `paraphrase-multilingual-MiniLM-L12-v2` embeddings) with sparse **Okapi BM25 keyword ranking**.
- Features preferred-category boosting and allergen restriction filtering.

### 4. 🧠 Zero-Hallucination State Anchor (`session_memory.py`)
- **Pronoun & Ordinal Resolution**: Resolves queries like *"add the 1st item"* or *"add it"* with display-order index snapshotting.
- **Response Syncing**: `sync_shown_items_from_response()` scans LLM text outputs to guarantee that whatever item is displayed on screen is 100% anchored in session memory.

### 5. 🪵 Live Terminal Diagnostic Logging (`log.py`)
- Color-coded real-time terminal logger reporting active LLM providers, vector encoding steps, hybrid search scores, intent classification, and interest score progression.

---

## 📁 Project Structure

```text
FoodieBot/
├── run_app.py              # Master launcher (Starts FastAPI + Streamlit in 1 command)
├── api.py                  # FastAPI REST API server (Port 8000) with Swagger UI
├── app.py                  # Streamlit Web UI application (Port 8501)
├── bot_logic.py            # Dual LLM Adapter (Gemini + Groq) & prompt rules
├── database.py             # ChromaDB vector DB, BM25 index & SentenceTransformers
├── session_memory.py       # State machine, pronoun resolution & response text syncing
├── ui_components.py        # Streamlit UI interface & hybrid search pipeline
├── interest_model.py       # Logistic Regression sentiment model (English/Hindi/Hinglish)
├── log.py                  # Centralized color-coded terminal logger engine
├── fast_food_products.csv  # Gourmet menu dataset (100 products)
├── requirements.txt        # Project Python dependencies
└── .env                    # Environment API keys
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/harshitttiwari/FoodieBot.git
cd FoodieBot

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows (or source venv/bin/activate on Linux/Mac)

# Install requirements
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY="your_gemini_api_key_here"
GROQ_API_KEY="your_groq_api_key_here"
```

### 3. Run Application (1-Command Master Launcher)

Launch both the **FastAPI REST Server** and **Streamlit Web UI** simultaneously:

```bash
python run_app.py
```

- 🌐 **Streamlit Web UI**: [http://localhost:8501](http://localhost:8501)
- 🌐 **Interactive Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- 🌐 **API Health Endpoint**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 📡 REST API Reference

### `POST /api/chat`
Sends a user message to FoodieBot and receives the bot response, action type, interest score, and cart state.

**Request**:
```json
{
  "user_input": "suggest something spicy for rainy weather",
  "session_id": "user_session_1"
}
```

**Response**:
```json
{
  "user_input": "suggest something spicy for rainy weather",
  "bot_response": "• Spicy Thai Fusion Pizza – $15.49...",
  "action": "VIEW_MENU",
  "interest_score": 58,
  "cart_changed": false,
  "cart_items_count": 0,
  "latency_ms": 620.45
}
```

---

### `GET /api/menu`
Retrieves menu items with optional category filtering and keyword search.

**Query Parameters**:
- `category` *(optional)*: Filter by category (e.g. `Pizza`, `Burgers`, `Beverages`).
- `search` *(optional)*: Filter by keyword in product name.

---

## 📊 Benchmark & Evaluation Metrics

- **Automated Integration Stress Suite**: **100% Pass Rate** across 14 multi-turn integration test cases.
- **Out-of-Sample Generalization**: **80.0% Overall Score** across 20 unseen multi-turn prompts.
- **Intent Parsing Accuracy**: **88.2%** on novel Hinglish, slang, and typo inputs.

---

## 📜 License

Distributed under the MIT License. Developed for Advanced AI Application Development.
