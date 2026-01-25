import streamlit as st
import os
import pandas as pd
import cohere
from pinecone import Pinecone
from dotenv import load_dotenv

# --------------------------------------------------
# App setup
# --------------------------------------------------
st.set_page_config(
    page_title="AI Agriculture Dashboard",
    page_icon="🌽",
    layout="wide"
)

st.title("🌽 AI-Powered Agriculture Decision Support")
st.caption("Cohere • Pinecone • RAG over real farm survey data")

# --------------------------------------------------
# Environment & clients
# --------------------------------------------------
load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Controls")
TOP_K = st.sidebar.slider("Top-K Retrieved Records", min_value=1, max_value=10, value=5)
SHOW_CONTEXT = st.sidebar.checkbox("Show retrieved context", value=True)

# --------------------------------------------------
# Dataset overview (placeholder – CSV stats)
# --------------------------------------------------
@st.cache_data
def load_csv(path: str):
    return pd.read_csv(path)

csv_path = st.sidebar.text_input("CSV path (optional)", value="")
if csv_path:
    try:
        df = load_csv(csv_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records", len(df))
        c2.metric("Counties", df['County'].nunique())
        c3.metric("Crops", df['Crop'].nunique())
        c4.metric("Avg Yield", round(df['Yield'].mean(), 2))
    except Exception as e:
        st.warning(f"Could not load CSV: {e}")

st.divider()

# --------------------------------------------------
# Retrieval + Generation
# --------------------------------------------------

def retrieve(query: str, top_k: int = 5):
    emb = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    query_vec = emb.embeddings.float[0]

    res = index.query(
        vector=query_vec,
        top_k=top_k,
        include_metadata=True,
    )
    return res.get('matches', [])


def build_prompt(question: str, matches):
    context_lines = []
    for i, m in enumerate(matches, start=1):
        meta = m.get('metadata', {})
        context_lines.append(
            f"[{i}] County={meta.get('county')} | Crop={meta.get('crop')} | Yield={meta.get('yield')}"
        )
    context = "\n".join(context_lines)

    prompt = f"""
You are an agriculture data assistant.
Use ONLY the context below to answer the question.
If the answer is not present, say you don't know.

Context:
{context}

Question: {question}
Answer:
"""
    return prompt

def generate_answer(prompt: str):

    st.subheader("Debug: final prompt sent to cohere")
    st.code(prompt)

    response = co.chat(
        model="command-a-03-2025",
        message=prompt,
        temperature=0,
        max_tokens=200,
    )
    return response.text

# --------------------------------------------------
# UI: Ask a question
# --------------------------------------------------
st.subheader("🔍 Ask the AI about the data")
question = st.text_input("Ask a question", placeholder="e.g., What is the average yield in County X?")

if st.button("Ask AI") and question:
    with st.spinner("Retrieving relevant records..."):
        matches = retrieve(question, TOP_K)

    prompt = build_prompt(question, matches)

    with st.spinner("Generating answer..."):
        answer = generate_answer(prompt)

    st.markdown("### 🤖 Answer")
    st.write(answer)

    if SHOW_CONTEXT:
        st.markdown("### 📄 Retrieved Context")
        for m in matches:
            st.json(m.get('metadata', {}))

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()
st.caption("Built by Pratham Goswami • AI + Agriculture")
