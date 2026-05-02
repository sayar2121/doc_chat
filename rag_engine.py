"""
rag_engine.py
Core RAG logic: PDF loading, chunking, embedding, FAISS storage, querying.
"""

import os
import pickle
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz
import google.generativeai as genai

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 4
EMBED_MODEL   = "all-MiniLM-L6-v2"
VECTOR_DIR    = "vector_store"
DB_PATH       = "chat_history.db"

print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL)
print("Model ready!")


def load_pdf(filepath):
    doc = fitz.open(filepath)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"text": text, "page": i + 1, "source": os.path.basename(filepath)})
    doc.close()
    return pages


def split_into_chunks(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            chunk_text = text[start: start + CHUNK_SIZE]
            if chunk_text.strip():
                chunks.append({"text": chunk_text, "page": page["page"], "source": page["source"]})
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_vector_store(chunks, store_name):
    os.makedirs(VECTOR_DIR, exist_ok=True)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    vectors = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    vectors = vectors.astype("float32")
    faiss.normalize_L2(vectors)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, os.path.join(VECTOR_DIR, f"{store_name}.index"))
    with open(os.path.join(VECTOR_DIR, f"{store_name}.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    print("Vector store saved!")


def load_vector_store(store_name):
    index_path = os.path.join(VECTOR_DIR, f"{store_name}.index")
    meta_path  = os.path.join(VECTOR_DIR, f"{store_name}.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No vector store for '{store_name}'. Upload the PDF first.")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


def list_available_stores():
    if not os.path.exists(VECTOR_DIR):
        return []
    return sorted({f.replace(".index", "") for f in os.listdir(VECTOR_DIR) if f.endswith(".index")})


def retrieve_chunks(query, index, chunks, top_k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def build_prompt(query, retrieved):
    context_parts = []
    for chunk in retrieved:
        context_parts.append(f"[Source: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    return f"""You are a helpful assistant with broad knowledge.

First, answer using the provided document context below.
Then, if you have additional knowledge beyond the document, add it under a section called "Additional Information" to give a more complete answer.
Clearly separate what came from the document vs your own knowledge.

Document Context:
{context}

Question: {query}

Answer:"""


def call_llm(prompt, api_key):
    from groq import Groq
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session TEXT, role TEXT, message TEXT, timestamp TEXT)""")
    conn.commit()
    conn.close()


def save_message(session, role, message):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history (session, role, message, timestamp) VALUES (?,?,?,?)",
                 (session, role, message, datetime.now().isoformat()))
    conn.commit()
    conn.close()


def load_history(session):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role, message FROM chat_history WHERE session=? ORDER BY id", (session,)).fetchall()
    conn.close()
    return [{"role": r[0], "message": r[1]} for r in rows]


def clear_history(session):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chat_history WHERE session=?", (session,))
    conn.commit()
    conn.close()


def ingest_pdf(filepath):
    store_name = os.path.splitext(os.path.basename(filepath))[0]
    store_name = "".join(c if c.isalnum() or c == "_" else "_" for c in store_name)
    pages  = load_pdf(filepath)
    chunks = split_into_chunks(pages)
    build_vector_store(chunks, store_name)
    return store_name


def answer_question(query, store_name, api_key, session):
    init_db()
    index, chunks = load_vector_store(store_name)
    retrieved     = retrieve_chunks(query, index, chunks)
    prompt        = build_prompt(query, retrieved)
    answer        = call_llm(prompt, api_key)
    save_message(session, "user", query)
    save_message(session, "assistant", answer)
    sources = list({f"{c['source']} (page {c['page']})" for c in retrieved})
    return {"answer": answer, "sources": sources, "retrieved_chunks": retrieved}