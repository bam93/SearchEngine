import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess

# ----------------------------
# Config
# ----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "mistral:7b"
TOP_K = 5
THRESHOLD_GOOD = 0.70
THRESHOLD_LOW = 0.40
SUMMARY_WEIGHT = 0.7
KEYWORDS_WEIGHT = 0.3

# Streamlit layout and style
st.set_page_config(page_title="🔍 AI Documentation Assistant", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .title {
        font-size: 2.2em;
        font-weight: 600;
        color: #374151;
        margin-bottom: 10px;
    }
    .score-bar {
        height: 16px;
        border-radius: 8px;
        background: linear-gradient(to right, #10b981, #3b82f6);
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------------
# Load models & collection
# ----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="web_chunks")
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------------
# Functions
# ----------------------------

def get_top_k_matches(question, k=TOP_K):
    query_embedding = embedder.encode([question])[0]
    all_docs = collection.get()

    summaries = [meta.get("summary", "") for meta in all_docs["metadatas"]]
    keywords_list = [", ".join(meta.get("keywords", [])) for meta in all_docs["metadatas"]]
    ids = all_docs["ids"]
    metadatas = all_docs["metadatas"]
    texts = all_docs["documents"]

    summary_embeddings = embedder.encode(summaries)
    keywords_embeddings = embedder.encode(keywords_list)

    results = []
    for i in range(len(summaries)):
        combined_embedding = SUMMARY_WEIGHT * summary_embeddings[i] + KEYWORDS_WEIGHT * keywords_embeddings[i]
        similarity = cosine_similarity([query_embedding], [combined_embedding])[0][0]

        results.append({
            "id": ids[i],
            "similarity": float(similarity),
            "summary": summaries[i],
            "text": texts[i],
            "metadata": metadatas[i]
        })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:k]

def generate_answer_with_llm(context, question):
    prompt = f"""Context:
{context}

Question: {question}

Give a concise and accurate answer, listing helpful links and keywords when relevant.
"""
    result = subprocess.run(["ollama", "run", LLM_MODEL], input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip() if result.returncode == 0 else f"LLM error: {result.stderr.decode('utf-8')}"

def generate_answer_without_context(question):
    prompt = f"""Answer the following question as best as you can, even without any context:

Question: {question}
"""
    result = subprocess.run(["ollama", "run", LLM_MODEL], input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.decode("utf-8").strip() if result.returncode == 0 else f"LLM error: {result.stderr.decode('utf-8')}"

# ----------------------------
# UI
# ----------------------------

st.markdown("<div class='title'>🤖 AI Documentation Search Assistant</div>", unsafe_allow_html=True)
query = st.text_input("💬 Ask your question:", placeholder="How do I configure a Python environment?")

if st.button("🔎 Run Search"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        top_docs = get_top_k_matches(query)
        best_similarity = top_docs[0]["similarity"]

        st.markdown("### 📄 Top Matching Documents")
        for doc in top_docs:
            similarity_pct = round(doc["similarity"] * 100, 2)

            with st.expander(f"📘 {doc['metadata']['title'] or 'Untitled Page'} — Similarity: {similarity_pct:.2f}%"):
                st.markdown(f"🔗 [Open Page]({doc['metadata']['url']})")
                st.markdown(f"**Summary:** {doc['summary']}")
                st.markdown(f"**Keywords:** {', '.join(doc['metadata'].get('keywords', []))}")
                st.progress(doc["similarity"])

        # LLM response section
        if best_similarity >= THRESHOLD_GOOD:
            st.success("✅ High-confidence match. Generating LLM response with context...")
            full_context = "\n\n".join([d["summary"] for d in top_docs])
            answer = generate_answer_with_llm(full_context, query)
            st.markdown("### 🧠 LLM Answer (High Confidence)")
            st.info(answer)

        elif best_similarity >= THRESHOLD_LOW:
            st.warning("⚠️ Medium-confidence match. You may still try the LLM with context.")
            if st.button("🛠 Generate LLM Answer Anyway"):
                full_context = "\n\n".join([d["summary"] for d in top_docs])
                answer = generate_answer_with_llm(full_context, query)
                st.markdown("### 🧠 LLM Answer (Medium Confidence)")
                st.info(answer)

        else:
            st.error("🚫 No relevant match found. Generating LLM response without context.")
            answer = generate_answer_without_context(query)
            st.markdown("### 🧠 LLM Answer (No Context)")
            st.info(answer)
