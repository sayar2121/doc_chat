# 📄 DocChat — RAG-Based Document Chatbot
Final year B.Tech Project | Computer Science | AI/ML

## 🚀 Run in 3 steps

```bash
# Step 1 — Install dependencies
pip install -r requirements.txt

# Step 2 — Add your Groq API key
# Open .streamlit/secrets.toml and paste your key

# Step 3 — Run
python -m streamlit run app.py
```

App opens at → http://localhost:8501

## 🔑 Get free Groq API key (100% free, no credit card)
1. Go to https://console.groq.com
2. Sign up with Google
3. Click "API Keys" → "Create API Key"
4. Copy the key (starts with gsk_...)
5. Paste it in `.streamlit/secrets.toml`

## 🏗️ How RAG works

```
PDF Upload → Text Chunking → Embedding (local, free)
                                    ↓
                             FAISS Vector Store
                                    ↓
User Question → Embed → Similarity Search → Top 4 Chunks
                                                  ↓
                                    Groq (Llama 3.3 70B) — free
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
│   ├── secrets.toml          ← Your Groq API key goes here
│   └── config.toml           ← Streamlit settings
├── uploads/                  ← PDFs saved here (auto-created)
├── vector_store/             ← FAISS index files (auto-created)
└── chat_history.db           ← SQLite DB (auto-created)
```

## 💰 Cost

- Embedding model : FREE (runs locally on your laptop)
- Groq Llama 3.3  : FREE (generous free tier, no card needed)
- FAISS           : FREE (local vector store)
- Streamlit       : FREE

**Total cost: ₹0**

## 🔧 Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| PDF Parsing | PyMuPDF | Extract text from PDFs |
| Text Splitting | Custom chunker | Split into 500-char overlapping chunks |
| Embedding | sentence-transformers (all-MiniLM-L6-v2) | Convert text to vectors locally |
| Vector Store | FAISS | Fast similarity search |
| LLM | Groq — Llama 3.3 70B | Generate answers (free) |
| Chat History | SQLite | Store conversations |
| UI | Streamlit | Web interface |

## 🆚 Why DocChat vs ChatGPT

| Feature | ChatGPT | DocChat |
|---------|---------|---------|
| Knowledge source | Internet training data | Your uploaded PDF only |
| Data privacy | Data sent to OpenAI | Runs locally, private |
| Hallucination | Can make up answers | Grounded in document |
| Source citation | No page numbers | Shows exact page number |
| Cost | Paid subscription | ₹0 completely free |
| Transparency | Black box | Shows retrieved chunks |

## 🎓 For your viva

- **Click "Show retrieved chunks"** — demonstrates exactly how RAG retrieval works
- **RAG vs Fine-tuning** — RAG doesn't change the model weights, it gives the model fresh context at query time
- **Why vector search** — embedding similarity is semantic (understands meaning), unlike keyword search which only matches exact words
- **Why FAISS** — stores millions of vectors and searches in milliseconds using approximate nearest neighbour algorithm
- **Privacy advantage** — embedding model runs 100% offline, only the final prompt goes to Groq API
