"""
app.py
Streamlit UI for the RAG Document Chatbot.
Run: streamlit run app.py
"""

import os
import uuid
import streamlit as st
from rag_engine import (
    ingest_pdf,
    answer_question,
    list_available_stores,
    load_vector_store,
    retrieve_chunks,
    clear_history,
    init_db,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "store_name" not in st.session_state:
    st.session_state.store_name = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    # Auto-load from secrets.toml if it exists
    try:
        st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        st.session_state.api_key = ""

init_db()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-bubble {
    background: #1a73e8; color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px; margin: 8px 0 8px 60px;
}
.bot-bubble {
    background: #f1f3f4; color: #202124;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px; margin: 8px 60px 8px 0;
}
.source-pill {
    font-size: 11px; color: #555;
    background: #e8eaf6; border-radius: 12px;
    padding: 2px 10px; margin: 4px 3px 0 0;
    display: inline-block;
}
.welcome-box {
    text-align: center; padding: 60px 20px;
    color: #888;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 DocChat")
    st.caption("RAG-powered document chatbot")
    st.divider()

    # ── API Key ──
    st.subheader("🔑 Groq API Key")
    key_input = st.text_input(
        "Paste your key here",
        type="password",
        value=st.session_state.api_key,
        placeholder="********************",
        help="Get free key at aistudio.google.com",
    )
    if key_input:
        st.session_state.api_key = key_input
    if st.session_state.api_key:
        st.success("✅ Key loaded")
    else:
        st.info("🔑 Paste your Gocq API key above")

    st.divider()

    # ── Upload PDF ──
    st.subheader("📂 Upload PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded:
        if st.button("⚙️ Process PDF", use_container_width=True, type="primary"):
            os.makedirs("uploads", exist_ok=True)
            save_path = os.path.join("uploads", uploaded.name)
            with open(save_path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Reading PDF and building index... (~30 sec)"):
                try:
                    store = ingest_pdf(save_path)
                    st.session_state.store_name = store
                    st.session_state.messages   = []
                    clear_history(st.session_state.session_id)
                    st.success(f"✅ Ready! '{uploaded.name}' indexed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()

    # ── Switch between indexed PDFs ──
    stores = list_available_stores()
    if stores:
        st.subheader("📚 Switch Document")
        selected = st.selectbox("Previously indexed PDFs", stores)
        if st.button("Load this PDF", use_container_width=True):
            st.session_state.store_name = selected
            st.session_state.messages   = []
            clear_history(st.session_state.session_id)
            st.rerun()

    st.divider()

    # ── Clear chat ──
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        clear_history(st.session_state.session_id)
        st.rerun()

    st.caption(f"Session ID: {st.session_state.session_id}")
    st.caption("Built with Gemini · FAISS · Streamlit")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("💬 Chat with your PDF")

# ── Guard: no doc or key ──
if not st.session_state.store_name and not st.session_state.api_key:
    st.markdown("""
    <div class="welcome-box">
        <h2>👋 Welcome to DocChat</h2>
        <p>1. Paste your <b>Gemini API key</b> in the sidebar</p>
        <p>2. Upload a <b>PDF file</b> and click Process</p>
        <p>3. Start asking questions!</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.api_key:
    st.warning("🔑 Please paste your Gemini API key in the sidebar.")
    st.stop()

if not st.session_state.store_name:
    st.info("📂 Upload and process a PDF from the sidebar to start chatting.")
    st.stop()

st.caption(f"📄 Active document: **{st.session_state.store_name}**")

# ── Chat messages ──
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        sources_html = "".join(
            f'<span class="source-pill">📌 {s}</span>'
            for s in msg.get("sources", [])
        )
        st.markdown(
            f'<div class="bot-bubble">🤖 {msg["content"]}'
            f'{"<br/><br/>" + sources_html if sources_html else ""}</div>',
            unsafe_allow_html=True,
        )

# ── Input form ──
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "question",
            label_visibility="collapsed",
            placeholder="Ask anything about your document...",
        )
    with col2:
        submitted = st.form_submit_button("Send ➤", use_container_width=True)

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("Searching document and generating answer..."):
        try:
            result = answer_question(
                query      = user_input,
                store_name = st.session_state.store_name,
                api_key    = st.session_state.api_key,
                session    = st.session_state.session_id,
            )
            st.session_state.messages.append({
                "role":    "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
        except Exception as e:
            st.session_state.messages.append({
                "role":    "assistant",
                "content": f"⚠️ Error: {str(e)}",
                "sources": [],
            })
    st.rerun()

# ── Retrieved chunks panel (for demo / viva) ──
if st.session_state.messages:
    last_user_msg = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        None,
    )
    if last_user_msg:
        with st.expander("🔍 Show retrieved chunks (for college demo / viva)"):
            try:
                index, chunks = load_vector_store(st.session_state.store_name)
                retrieved     = retrieve_chunks(last_user_msg, index, chunks)
                for i, chunk in enumerate(retrieved, 1):
                    st.markdown(
                        f"**Chunk {i}** — `{chunk['source']}` page {chunk['page']} "
                        f"| similarity score: `{chunk['score']:.3f}`"
                    )
                    st.text_area(
                        f"chunk_{i}",
                        value=chunk["text"][:400] + "...",
                        height=100,
                        label_visibility="collapsed",
                    )
            except Exception as e:
                st.error(f"Could not load chunks: {e}")
