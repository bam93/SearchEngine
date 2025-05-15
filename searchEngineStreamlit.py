# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# File        : searchEngineDash.py
# Description : Dash interface for conversational search using ChromaDB and LLMs
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.1 (filtered + correct cosine weighting)
# -----------------------------------------------------------------------------

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import json
import subprocess
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from chromadb import PersistentClient

# ----------------------------
# Configuration
# ----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "deepseek-r1:14b"
TOP_K = 5
THRESHOLD_GOOD = 0.70
THRESHOLD_LOW = 0.40
SUMMARY_WEIGHT = 0.8
KEYWORDS_WEIGHT = 0.2

# ----------------------------
# Load Chroma Vector Store
# ----------------------------
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="web_chunks")

# ----------------------------
# Embedding utility (external script via subprocess)
# ----------------------------
def embed_texts(texts):
    try:
        result = subprocess.run(
            ["python", "embed_worker.py"],
            input=json.dumps(texts).encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return np.array(json.loads(result.stdout.decode("utf-8")))
    except Exception as e:
        return np.zeros((len(texts), 384))

# ----------------------------
# LLM utility
# ----------------------------
def call_ollama_llm(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", LLM_MODEL],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        return f"LLM error: {e.stderr.decode('utf-8')}"

# ----------------------------
# Core similarity logic with page deduplication
# ----------------------------
def get_top_k_matches(question):
    query_embedding = embed_texts([question])[0]
    all_docs = collection.get()

    summaries = [meta.get("summary", "") for meta in all_docs["metadatas"]]
    keywords_list = [", ".join(meta.get("keywords", [])) for meta in all_docs["metadatas"]]
    web_paths = [meta.get("web_path", "") for meta in all_docs["metadatas"]]

    summary_embeddings = embed_texts(summaries)
    keyword_embeddings = embed_texts(keywords_list)

    results = []
    for i in range(len(summaries)):
        combined_embedding = (SUMMARY_WEIGHT * summary_embeddings[i] + KEYWORDS_WEIGHT * keyword_embeddings[i]) / 1.0
        similarity = cosine_similarity([query_embedding], [combined_embedding])[0][0]
        results.append({
            "id": all_docs["ids"][i],
            "similarity": float(similarity),
            "summary": summaries[i],
            "text": all_docs["documents"][i],
            "metadata": all_docs["metadatas"][i],
            "web_path": web_paths[i]
        })

    sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    # Filter to only the best chunk per unique page
    seen_paths = set()
    unique_results = []
    for doc in sorted_results:
        if doc["web_path"] not in seen_paths:
            seen_paths.add(doc["web_path"])
            unique_results.append(doc)
        if len(unique_results) >= TOP_K:
            break

    return unique_results

# ----------------------------
# Dash Layout
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    html.H1("🔍 Conversational Search Over Documentation"),
    dcc.Input(id="query-input", type="text", placeholder="Ask a question...", style={"width": "100%"}),
    html.Br(), html.Br(),
    dbc.Button("Run Search", id="search-btn", color="primary"),
    html.Hr(),
    html.Div(id="results-output")
])

@app.callback(
    Output("results-output", "children"),
    Input("search-btn", "n_clicks"),
    State("query-input", "value")
)
def update_output(n_clicks, query):
    if not n_clicks or not query or not query.strip():
        return dbc.Alert("Please enter a valid question.", color="warning")

    matches = get_top_k_matches(query)
    if not matches:
        return dbc.Alert("No documents found.", color="danger")

    best_score = matches[0]['similarity']
    content = []

    for doc in matches:
        similarity_pct = round(doc["similarity"] * 100, 2)
        content.extend([
            html.H5(html.A("🔗 View Page", href=doc["metadata"]["url"], target="_blank")),
            html.P(f"Summary: {doc['summary']}", style={"margin-bottom": "4px"}),
            html.P(f"Keywords: {doc['metadata'].get('keywords', '')}"),
            html.P(f"Similarity Score: {similarity_pct} %"),
            html.Hr()
        ])

    if best_score >= THRESHOLD_GOOD:
        context = "\n\n".join([d["summary"] for d in matches])
        llm_output = call_ollama_llm(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        content.append(html.Div([
            html.H4("🧠 LLM Response (with context):"),
            html.Pre(llm_output)
        ]))

    elif best_score >= THRESHOLD_LOW:
        context = "\n\n".join([d["summary"] for d in matches])
        llm_output = call_ollama_llm(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:")
        content.append(html.Div([
            html.H4("🧠 LLM Response (medium confidence):"),
            html.Pre(llm_output)
        ]))

    else:
        llm_output = call_ollama_llm(f"Answer the following question as best you can:\n\nQuestion: {query}")
        content.append(html.Div([
            html.H4("🧠 LLM Response (no context):"),
            html.Pre(llm_output)
        ]))

    return content

if __name__ == '__main__':
    app.run(debug=True)
