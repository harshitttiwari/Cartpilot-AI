# 🤖 FoodieBot - AI Restaurant Assistant

An intelligent restaurant chatbot powered by RAG (Retrieval-Augmented Generation) that helps customers explore menu items with real-time search and personalized recommendations.

## ✨ Features

- **Smart Menu Search**: Natural language queries to find food items
- **RAG Pipeline**: ChromaDB + SentenceTransformers for accurate retrieval
- **LLM Integration**: Groq API with Llama-3.1-8b-instant for conversational responses
- **Live Analytics**: Real-time query tracking and interest scoring
- **Admin Panel**: Edit menu data directly in the interface
- **Fixed Chat UI**: 350px scrollable chat history for optimal UX

## 🚀 Quick Deploy

### Hugging Face Spaces (Recommended)
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → New Space
2. Name: `foodiebot`, SDK: `Streamlit`, Python: `3.10`
3. Add files from this repo (or import from GitHub: `harshitttiwari/FoodieBot`)
4. Set Environment Variable: `GROQ_API_KEY` = `your_groq_api_key`
5. Deploy automatically

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
COPY . .
EXPOSE 7860
CMD ["sh", "-c", "streamlit run app.py --server.port=$PORT --server.address=0.0.0.0"]
```

### Local Setup
```bash
git clone https://github.com/harshitttiwari/FoodieBot.git
cd FoodieBot
pip install -r requirements.txt
echo 'GROQ_API_KEY="your_key_here"' > .env
streamlit run app.py
```

## 📁 Project Structure

```
FoodieBot/
├── app.py                 # Main Streamlit application
├── bot_logic.py          # LLM integration and response logic
├── database.py           # ChromaDB setup and embeddings
├── ui_components.py      # Chat interface and analytics
├── fast_food_products.csv # Menu data
├── requirements.txt      # Dependencies
└── .env                  # API keys (local only)
```

## 🔧 Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for LLM access
- `HF_HOME`: Hugging Face cache directory (auto-set to `/tmp/hf_cache`)
- `TRANSFORMERS_CACHE`: Transformers cache (auto-set to `/tmp/hf_cache`)

### Dependencies
- streamlit==1.36.0
- python-dotenv==1.0.1
- langchain-groq==0.1.3
- pandas==2.2.2
- chromadb==0.5.3
- sentence-transformers==2.2.2
- torch==2.1.2

## 🎯 Usage

1. **Ask Questions**: "Show me spicy burgers under $12"
2. **Get Recommendations**: "What's good for someone with dairy allergies?"
3. **View Analytics**: Check query performance and interest scores
4. **Admin Access**: Edit menu items in the sidebar panel

## 🔒 Security Notes

- Never commit `.env` files (already gitignored)
- Use environment variables or platform secrets for API keys
- Rotate exposed API keys immediately

## 🐛 Troubleshooting

**Permission denied: '/app/model_cache'**
- Fixed: App sets `HF_HOME=/tmp/hf_cache` for writable cache

**ModuleNotFoundError: dotenv**
- Fixed: dotenv import is optional; uses platform secrets as fallback

**Chat area too large**
- Fixed: Chat history constrained to 350px scrollable container

## 📊 Architecture

```
User Query → ChromaDB Search → Context Building → Groq LLM → Formatted Response
     ↓              ↓                ↓              ↓            ↓
Analytics ← Query Logging ← Interest Scoring ← Response ← Clean Formatting
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature-name`
5. Submit a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Live Demo](https://huggingface.co/spaces/harshitttiwari/foodiebot)
- [GitHub Repository](https://github.com/harshitttiwari/FoodieBot)
- [Groq API](https://groq.com/)
