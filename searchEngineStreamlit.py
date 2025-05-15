# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# File        : searchEngineDash.py
# Description : Dash interface for conversational search with filters
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.3 (Filtering by path, keyword, free text)
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
SUMMARY_WEIGHT = 0.8
KEYWORDS_WEIGHT = 0.2
LLM_MODEL = "deepseek-r1:14b"
TOP_K = 10
THRESHOLD_GOOD = 0.70
SUMMARY_DISPLAY = 3

# ----------------------------
# Load Chroma Vector Store
# ----------------------------
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="web_chunks")
all_docs = collection.get()

# Extract filter options
all_paths = sorted(set(m.get("web_path", "") for m in all_docs["metadatas"]))
all_keywords = sorted({k for m in all_docs["metadatas"] for k in m.get("keywords", [])})

# ----------------------------
# Embedding utility
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
    except Exception:
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
# Dash Layout
# ----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return;
        const text = document.getElementById("llm-answer")?.innerText || "";
        navigator.clipboard.writeText(text);
        alert("Copied to clipboard!");
    }
    """,
    Output("copy-btn", "n_clicks"),
    Input("copy-btn", "n_clicks")
)

app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return;
        const element = document.getElementById("llm-answer");
        if (!element) return;
        const content = element.innerText;
        const win = window.open("", "_blank");
        win.document.write(`<pre>${content}</pre>`);
        win.document.close();
        win.print();
    }
    """,
    Output("export-pdf-btn", "n_clicks"),
    Input("export-pdf-btn", "n_clicks")
)

app.layout = dbc.Container([
    html.H1("🔍 Conversational Search Over Documentation"),
    dbc.Row([
        dbc.Col(dcc.Input(id="query-input", type="text", placeholder="Ask a question...", style={"width": "100%"}), md=8),
        dbc.Col(dbc.Button("Run Search", id="search-btn", color="primary", className="mt-1"), md=4),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Label("Section (web path):"),
            dcc.Dropdown(id="path-filter", options=[{"label": p, "value": p} for p in all_paths], multi=False)
        ], md=6),
        dbc.Col([
            html.Label("Keyword tag:"),
            dcc.Dropdown(id="keyword-filter", options=[{"label": k, "value": k} for k in all_keywords], multi=False)
        ], md=6),
    ]),
    html.Br(),
    html.Label("Free text filter (within document text):"),
    dcc.Input(id="free-text", type="text", placeholder="Contains...", style={"width": "100%"}),
    html.Hr(),
    html.Div(id="results-output")
])

# ----------------------------
# Main callback
# ----------------------------
@app.callback(
    Output("results-output", "children"),
    Input("search-btn", "n_clicks"),
    State("query-input", "value"),
    State("path-filter", "value"),
    State("keyword-filter", "value"),
    State("free-text", "value")
)
def update_output(n_clicks, query, path_val, keyword_val, free_val):
    if not n_clicks or not query or not query.strip():
        return dbc.Alert("Please enter a valid question.", color="warning")

    # Filter docs based on criteria
    filtered = []
    for i, doc in enumerate(all_docs["documents"]):
        meta = all_docs["metadatas"][i]
        if path_val and meta.get("web_path") != path_val:
            continue
        if keyword_val and keyword_val not in meta.get("keywords", []):
            continue
        if free_val and free_val.lower() not in doc.lower():
            continue
        filtered.append({
            "id": all_docs["ids"][i],
            "text": doc,
            "metadata": meta
        })

    if not filtered:
        return dbc.Alert("No documents match the selected filters.", color="danger")

    query_embedding = embed_texts([query])[0]
    summaries = [f["metadata"].get("summary", "") for f in filtered]
    keywords = [", ".join(f["metadata"].get("keywords", [])) for f in filtered]
    summary_embeddings = embed_texts(summaries)
    keyword_embeddings = embed_texts(keywords)

    results = []
    for i in range(len(filtered)):
        combined_embedding = (SUMMARY_WEIGHT * summary_embeddings[i] + KEYWORDS_WEIGHT * keyword_embeddings[i])
        sim = cosine_similarity([query_embedding], [combined_embedding])[0][0]
        results.append({
            "id": filtered[i]["id"],
            "text": filtered[i]["text"],
            "metadata": filtered[i]["metadata"],
            "similarity": float(sim)
        })

    
    

    top_matches = sorted(results, key=lambda x: x["similarity"], reverse=True)[:TOP_K]

    # Display summaries
    content = []
    seen_paths = set()
    display_matches = []
    for doc in top_matches:
        path = doc["metadata"].get("web_path")
        if path not in seen_paths:
            seen_paths.add(path)
            display_matches.append(doc)
        if len(display_matches) >= SUMMARY_DISPLAY:
            break

    for doc in display_matches:
        content.extend([
            html.H5(html.A("🔗 View Page", href=doc["metadata"]["url"], target="_blank")),
            html.P(f"Summary: {doc['metadata'].get('summary', '')}"),
            html.P(f"Keywords: {', '.join(doc['metadata'].get('keywords', []))}"),
            html.P(f"Similarity Score: {round(doc['similarity'] * 100, 2)} %"),
            html.Hr()
        ])

    # Generate LLM response from all high-confidence pages
    high_confidence = [doc for doc in top_matches if doc['similarity'] >= THRESHOLD_GOOD]
    if high_confidence:
        all_context = "\n\n".join(doc['text'] for doc in high_confidence).join(doc['text'] for doc in high_confidence)
        urls = set(doc['metadata'].get('url') for doc in high_confidence)
        url_list = "\n".join(f"- {u}" for u in urls).join(f"- {u}" for u in urls)
        prompt = f"""
You are an expert assistant helping users understand official documentation.

The answer must be based only on the following documentation pages:
{url_list}

Documentation:
{all_context}

Question: {query}

Write a complete, structured, and practical answer. Include examples or code if applicable.
"""
        content.append(html.H5("📚 Pages used for this answer:"))
        content.append(html.Ul([html.Li(html.A(url, href=url, target="_blank")) for url in urls]))
        answer = call_ollama_llm(prompt)
        content.append(html.H4("🧠 LLM Answer (from all confident results):"))
        content.append(html.Pre(answer, id="llm-answer"))
        content.append(html.Div([
            html.Button("📋 Copy to clipboard", id="copy-btn", n_clicks=0, style={"marginRight": "10px"}),
            html.Button("🖨️ Export to PDF", id="export-pdf-btn", n_clicks=0)
        ]))

    return content

if __name__ == '__main__':
    app.run(debug=True)
