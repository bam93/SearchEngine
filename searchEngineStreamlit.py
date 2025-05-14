# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# File        : searchEngineStreamlit.py
# Description : Streamlit interface for conversational search using ChromaDB and LLMs
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.0
#
# This script launches a web-based interface for querying a vector store of
# enriched documentation (produced by generateRAG.py). It supports:
#
# - Natural language queries by the user
# - Semantic search over stored document summaries and keywords
# - Cosine similarity ranking using sentence-transformer embeddings
# - Three-level confidence strategy:
#     • High: contextual LLM answer from top-matching summaries
#     • Medium: prompt user to invoke LLM manually
#     • Low: fallback LLM answer without context
# - Integration with local Ollama LLM for fast response generation
#
# The interface is designed for researchers, engineers, or institutional users
# to explore and question technical documentation conversationally.
# -----------------------------------------------------------------------------


import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import subprocess

# ----------------------------
# Configuration
# ----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "mistral:7b"
TOP_K = 5
THRESHOLD_GOOD = 0.70
THRESHOLD_LOW = 0.40
TEMPERATURE = 0.1
SUMMARY_WEIGHT = 0.7
KEYWORDS_WEIGHT = 0.3

# ----------------------------
# Load Chroma Vector Store
# ----------------------------
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="web_chunks")

# ----------------------------
# Load Embedding Model
# ----------------------------
embedder = SentenceTransformer(EMBEDDING_MODEL)

# ----------------------------
# Functions
# ----------------------------

def get_top_k_matches(question, k=TOP_K):
    """Return top-k most similar documents based on weighted embedding (summary + keywords)"""
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
        # Weighted sum of summary and keyword embeddings
        combined_embedding = SUMMARY_WEIGHT * summary_embeddings[i] + KEYWORDS_WEIGHT * keywords_embeddings[i]
        similarity = cosine_similarity([query_embedding], [combined_embedding])[0][0]

        results.append({
            "id": ids[i],
            "similarity": float(similarity),
            "summary": summaries[i],
            "text": texts[i],
            "metadata": metadatas[i]
        })

    # Sort by similarity score (descending)
    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return sorted_results[:k]

def generate_answer_with_llm(context, question):
    """Query the LLM with provided context"""
    prompt = f"""Context:
{context}

Question: {question}

Give a concise and accurate answer, listing helpful links and keywords when relevant.
"""
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return f"LLM error: {result.stderr.decode('utf-8')}"
    return result.stdout.decode("utf-8").strip()

def generate_answer_without_context(question):
    """Query the LLM without any context (fallback for low similarity)"""
    prompt = f"""Answer the following question as best as you can, even without any context:

Question: {question}
"""
    result = subprocess.run(
        ["ollama", "run", LLM_MODEL],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if result.returncode != 0:
        return f"LLM error: {result.stderr.decode('utf-8')}"
    return result.stdout.decode("utf-8").strip()

# ----------------------------
# Streamlit User Interface
# ----------------------------
st.set_page_config(page_title="🔍 RAG Document Search", layout="wide")
st.title("🔍 Conversational Search Over Documentation")

# User input
query = st.text_input("Ask a question:", placeholder="How do I configure a Python environment?")

if st.button("🔎 Run Search"):
    if not query.strip():
        st.warning("Please enter a valid question.")
    else:
        st.info("Searching the vector database for relevant content...")
        top_docs = get_top_k_matches(query)

        best_similarity = top_docs[0]["similarity"]

        # Display top document matches
        st.subheader("🔎 Top Matching Documents")
        for doc in top_docs:
            similarity_pct = round(doc["similarity"] * 100, 2)
            st.markdown(f"**🔗 [View Page]({doc['metadata']['url']})**")
            st.markdown(f"**Summary:** {doc['summary']}")
            st.markdown(f"**Keywords:** {', '.join(doc['metadata'].get('keywords', []))}")
            st.markdown(f"**Similarity Score:** {similarity_pct} %")
            st.markdown("---")

        # High similarity: use context + LLM
        if best_similarity >= THRESHOLD_GOOD:
            st.success("🎯 High confidence match. Generating a contextual LLM answer...")
            full_context = "\n\n".join([d["summary"] for d in top_docs])
            llm_response = generate_answer_with_llm(full_context, query)
            st.subheader("🧠 LLM Response (with context):")
            st.write(llm_response)

        # Medium similarity: suggest using LLM
        elif best_similarity >= THRESHOLD_LOW:
            st.warning("⚠️ Medium confidence. You can still ask the LLM to generate a response.")
            if st.button("🛠️ Generate Answer Anyway"):
                full_context = "\n\n".join([d["summary"] for d in top_docs])
                llm_response = generate_answer_with_llm(full_context, query)
                st.subheader("🧠 LLM Response (medium confidence):")
                st.write(llm_response)

        # Low similarity: fallback to LLM only
        else:
            st.error("🔴 Low similarity (< 40%). No good matches found. Generating an answer without context.")
            llm_response = generate_answer_without_context(query)
            st.subheader("🧠 LLM Response (no context):")
            st.write(llm_response)