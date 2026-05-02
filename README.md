# 📄 DocChat — RAG-Based Document Chatbot
 Computer Science | AI/ML

## 🚀 Run in 3 steps

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Add your Gemini API key
# Open .streamlit/secrets.toml and paste your key

# Step 3 — Run
streamlit run app.py
```

App opens at → http://localhost:8501

## 🔑 Get free Gemini API key
1. Go to https://aistudio.google.com
2. Sign in with Google
3. Click "Get API Key" → Create key
4. Paste it in `.streamlit/secrets.toml`

## 🏗️ How RAG works
```
PDF Upload → Text Chunking → Embedding (local, free)
                                    ↓
                             FAISS Vector Store
                                    ↓
User Question → Embed → Similarity Search → Top 4 Chunks
                                                  ↓
                                    Gemini 1.5 Flash (free)
                                                  ↓
                                      Answer + Page Citations
```

## 📂 Project files
```
rag_chatbot/
├── app.py                    ← Streamlit UI
├── rag_engine.py             ← Core RAG logic
├── requirements.txt          ← Python packages
├── .gitignore
├── .streamlit/
│   └── secrets.toml          ← Your API key goes here
├── uploads/                  ← PDFs saved here (auto-created)
├── vector_store/             ← FAISS index files (auto-created)
└── chat_history.db           ← SQLite DB (auto-created)
```

## 💰 Cost
- Embedding model: FREE (runs on your laptop)
- Gemini 1.5 Flash: FREE (60 requests/min free tier)
- FAISS: FREE (local)
- Streamlit: FREE

**Total cost: ₹0**

## 🎓 For your viva
- Click "Show retrieved chunks" panel — demonstrates how RAG retrieval works
- Explain: RAG vs fine-tuning (RAG doesn't change the model, just gives it context)
- Explain: why embeddings + vector search is faster than keyword search
