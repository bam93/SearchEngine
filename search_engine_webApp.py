# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# Organization: Centrale Méditerranée
# File        : search_engine_webApp.py
# Description : Supports RAG-only, LLM-only, and hybrid query modes.
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.3 (Interactive frontend built with Dash and Dash Bootstrap Components.)
#
# A Dash-based web application that implements a Retrieval-Augmented Generation (RAG) assistant using ChromaDB,
# custom embedding via a subprocess, and a local LLM (served via Ollama).
# -----------------------------------------------------------------------------

import json
import subprocess
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from chromadb import PersistentClient
from xhtml2pdf import pisa
from markdown2 import markdown
import tempfile
import base64
import sys
import requests

# --- Configuration ---
TOP_K = 50
THRESHOLD_GOOD = 0.70
DEFAULT_LLM_MODEL = "gemma3:4b"
DEFAULT_LANGUAGE = "EN"
DEFAULT_QUERY_MODE = "rag_only"

# --- Load Chroma Collection ---
client = PersistentClient(path="./chroma_db")
collection = client.get_collection(name="web_chunks")

# --- Embedding Utility ---
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
        print("❌ Embedding failed:", e)
        return np.zeros((len(texts), 384))

# --- LLM Call ---
import json

def call_ollama_llm(prompt, model, temperature=0.1):
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "options": {"temperature": temperature}
        }
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        if response.status_code == 200:
            answer_parts = []
            for line in response.iter_lines():
                if line:
                    try:
                        json_line = json.loads(line.decode("utf-8"))
                        if "response" in json_line:
                            answer_parts.append(json_line["response"])
                    except json.JSONDecodeError as e:
                        print("⚠️ JSON decode error in line:", line)
            return ''.join(answer_parts)
        else:
            return f"LLM API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"LLM API exception: {str(e)}"

# --- RAG Logic ---
# --- Traductions pour l'interface ---
def get_translations(lang):
    if lang == "FR":
        return {
            "ask_placeholder": "Posez votre question...",
            "model_label": "Modèle",
            "language_label": "Langue",
            "mode_label": "Mode « Assistant »",
            "mode_tooltip": "Choisissez comment l'assistant doit répondre :\n- Hybride : RAG avec complément LLM\n- RAG seul : uniquement depuis les documents\n- LLM seul : uniquement les connaissances du LLM",
            "submit": "Soumettre",
            "clear": "Effacer",
            "show_sources": "Afficher les sources",
            "you": "🧑 Vous",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources utilisées :",
            "download_pdf": "📄 Télécharger PDF",
            "no_question": "❗ Veuillez entrer une question.",
            "no_relevant_docs": "⚠️ Aucun document pertinent trouvé."
        }
    else:
        return {
            "ask_placeholder": "Ask your question...",
            "model_label": "Model",
            "language_label": "Language",
            "mode_label": "Mode",
            "mode_tooltip": "Choose how the assistant should answer:\n- Hybrid: RAG with LLM enhancement\n- RAG Only: answer only from documents\n- LLM Only: use only the LLM's internal knowledge",
            "submit": "Submit",
            "clear": "Clear Output",
            "show_sources": "Show sources",
            "you": "🧑 You",
            "assistant": "🤖 Assistant",
            "sources_used": "Sources used:",
            "download_pdf": "📄 Download PDF",
            "no_question": "❗ Please enter a question.",
            "no_relevant_docs": "⚠️ No relevant documents found."
        }

# --- Adaptation dynamique du prompt ---
def process_query(user_question, llm_model, lang, mode=DEFAULT_QUERY_MODE):
    temperature = 0.1 if mode == "rag_only" else 0.4 if mode == "hybrid" else 0.7

    translations = get_translations(lang)
 
#if lang == "FR":
#    intro = "Vous êtes un assistant expert qui aide les utilisateurs à comprendre de la documentation technique."
#    instruction_rag = "Fournissez une réponse détaillée, structurée et pratique uniquement à partir de la documentation fournie."
#    instruction_hybrid = "Ajoutez des
#     compléments issus du LLM si pertinent, en les identifiant clairement."
#else:
#    intro = "You are an expert assistant helping users understand technical documentation."
#    instruction_rag = "Provide a detailed, structured, and practical answer using only the provided documentation."
#    instruction_hybrid = "If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM."

    if lang == "FR":
        intro = "Vous êtes un assistant administratif expert qui aide les utilisateurs à comprendre la documentation de la grande école d'ingénieur Centrale Méditerranée."
        instruction_rag = "Fournissez une réponse détaillée dans un style académique, structurée et pratique uniquement à partir de la documentation fournie."
        instruction_hybrid = "Ajoutez des compléments issus du LLM si pertinent, en les identifiant clairement."
    else:
        intro = "You are an expert administrative assistant who helps users understand the documentation of the Big Éole d'Ingénieur Centrale Méditerranée."
        instruction_rag = "Provide a detailed, structured, and practical answer in an academic style using only the provided documentation."
        instruction_hybrid = "If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM."
    if mode == "llm_only":
        prompt = f"""
{intro}

Question: {user_question}

Answer strictly using the LLM's internal knowledge.
"""
        return call_ollama_llm(prompt, llm_model, temperature=temperature), []

    query_emb = embed_texts([user_question])[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    # Filtrage strict des documents pertinents
    relevant_data = [
        (doc, meta, score)
        for doc, meta, score in zip(docs, metas, scores)
        if score >= THRESHOLD_GOOD
    ]

    if not relevant_data:
        return translations["no_relevant_docs"], []

    page_map = {}
    for doc, meta, score in relevant_data:
        url = meta.get("url", "")
        if url not in page_map:
            page_map[url] = {"text": doc, "score": round(score, 4)}
        else:
            page_map[url]["text"] += "\n" + doc

    page_contexts = [
        {"url": url, "text": data["text"], "score": data["score"]}
        for url, data in page_map.items()
    ]

    all_text = "\n\n".join(p["text"] for p in page_contexts)
    all_urls = [p["url"] for p in page_contexts]

    prompt = f"""
{intro}

Sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation:
{all_text}

Question: {user_question}

{instruction_rag}
"""
    if mode == "hybrid":
        prompt += f"\n{instruction_hybrid}"

    return call_ollama_llm(prompt, llm_model, temperature=temperature), page_contexts

# --- PDF with markdown rendering ---
def generate_pdf(content, lang):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        html_template = f"""
        <html>
        <head>
            <meta charset='UTF-8'>
            <style>
                body {{ font-family: Helvetica, sans-serif; line-height: 1.4; font-size: 12pt; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                code {{ background-color: #f4f4f4; padding: 2px 4px; }}
            </style>
        </head>
        <body>{markdown(content)}</body>
        </html>
        """
        pisa_status = pisa.CreatePDF(html_template, dest=tmp_file)
        tmp_file.seek(0)
        pdf_data = tmp_file.read()
    return html.A(
        get_translations(lang)["download_pdf"],
        href=f"data:application/pdf;base64,{base64.b64encode(pdf_data).decode('utf-8')}",
        download="rag_answer.pdf",
        target="_blank",
        className="btn btn-outline-info mt-3"
    )
    
# --- Dash App Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "RAG Assistant"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🤖 RAG Assistant", className="text-primary"), width=8),
        dbc.Col(html.Img(src="/assets/logo_centrale.svg", height="60px"), 
        width=4, 
        style={"textAlign": "right"})
    ], align="center"),

    html.Hr(),

    dbc.Row([
        dbc.Col(dbc.Textarea(id="question-input", placeholder="Ask your question...", style={"height": "120px"}, className="mb-2"), width=8),
        dbc.Col([
            html.Label("Model", className="text-info fw-bold"),
            dbc.Select(
                id="llm-selector",
                options=[
                    {"label": "Gemma 3 (4b)", "value": "gemma3:4b"},
                    {"label": "Mistral 7B", "value": "mistral:7b"}
                ],
                value=DEFAULT_LLM_MODEL,
                className="mb-3"
            ),
            html.Label("Language", className="text-info fw-bold"),
            dbc.Select(
                id="lang-selector",
                options=[
                    {"label": "English", "value": "EN"},
                    {"label": "Français", "value": "FR"}
                ],
                value=DEFAULT_LANGUAGE,
                className="mb-3"
            ),
            html.Label([
                "Mode ",
                html.Span(
                    "ⓘ",
                    id="mode-tooltip",
                    style={"textDecoration": "underline dotted", "cursor": "pointer"},
                    title="Choose how the assistant should answer:\n- Hybrid: RAG with LLM enhancement\n- RAG Only: answer only from documents\n- LLM Only: use only the LLM's internal knowledge"
                )
            ], className="text-info fw-bold"),

            dbc.Select(
                id="mode-selector",
                options=[
                    {"label": "Hybrid (RAG + LLM)", "value": "hybrid"},
                    {"label": "RAG Only", "value": "rag_only"},
                    {"label": "LLM Only", "value": "llm_only"}
                ],
                value=DEFAULT_QUERY_MODE,
                className="mb-3"
            )
        ], width=4)
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit-button", color="success"), width="auto"),
        dbc.Col(dbc.Button("🧹 Clear Output", id="clear-button", color="warning", className="ms-2"), width="auto"),
        dbc.Col(dbc.Checkbox(id="show-sources-toggle", value=True, className="ms-3"), width="auto"),
        dbc.Col(html.Label("Show sources", className="mt-2"), width="auto")
    ], className="my-3", align="center"),

    dcc.Loading(
        id="loading-output",
        type="circle",  # or "default", "dot"
        color="#00ff99",  # optional: you can change the spinner color
        children=[
            dbc.Card([html.Div(id="chat-history", children=[], style={"margin": "10px"})], color="dark", inverse=True),
            html.Div(id="pdf-download", className="mt-3 text-end")
        ]
    ),
    dcc.Store(id="clear-question", data="")
], fluid=True, className="p-4", style={"backgroundColor": "#1e1e1e"})


@app.callback(
    Output("chat-history", "children"),
    Output("pdf-download", "children"),
    Output("question-input", "value"),
    Input("submit-button", "n_clicks"),
    Input("clear-button", "n_clicks"),
    State("question-input", "value"),
    State("show-sources-toggle", "value"),
    State("llm-selector", "value"),
    State("lang-selector", "value"),
    State("mode-selector", "value"),
    State("chat-history", "children"),
    prevent_initial_call=True
)
def update_chat(submit_clicks, clear_clicks, question, show_sources, llm_model, lang, mode, history):
    triggered_id = ctx.triggered_id
    if triggered_id == "clear-button":
        return [], "", ""

    if not question:
        return history + [html.Div("❗ Please enter a question.")], "", question

    answer, source_data = process_query(question, llm_model, lang, mode)
    formatted_answer = dcc.Markdown(answer)

    # Trier les sources par score décroissant (plus pertinent en premier)
    sorted_sources = sorted(source_data, key=lambda x: x["score"], reverse=True)

    if show_sources and sorted_sources:
        source_block = dcc.Markdown("\n".join([f"- {item['url']} (score: {item['score']})" for item in sorted_sources]))
    else:
        source_block = ""

    pdf_sources = "\n".join([f"- {item['url']} (score: {item['score']})" for item in sorted_sources])
    pdf_content = f"You: {question}\n\nAnswer:\n{answer}\n\nSources:\n{pdf_sources}"

    download_link = generate_pdf(pdf_content, lang)

    new_exchange = html.Div([
        html.H5("🧑 You:", className="text-warning"),
        html.Div(question, style={"whiteSpace": "pre-wrap", "marginBottom": "10px"}),
        html.H5("🤖 Assistant:", className="text-success"),
        html.Div(formatted_answer, style={
            "backgroundColor": "#2a2a2a",  # A bit lighter than black
            "padding": "10px",
            "borderRadius": "10px",
            "marginBottom": "10px"
        }),
        html.Div([html.Strong("Sources used:"), source_block], style={
            "marginTop": "10px",
            "color": "#ccc",  # Lighter gray
            "fontSize": "0.85em",
            "backgroundColor": "#1e1e1e",
            "padding": "8px",
            "borderRadius": "6px"
        })
    ], style={"marginBottom": "30px"})

    return history + [new_exchange], download_link, ""

if __name__ == "__main__":
    app.run(debug=True)
