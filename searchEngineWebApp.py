import json
import subprocess
import numpy as np
from dash import Dash, html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc
from chromadb import PersistentClient
import markdown
from xhtml2pdf import pisa
import base64
import tempfile

# --- Configuration ---
TOP_K = 50
THRESHOLD_GOOD = 0.70
DEFAULT_LLM_MODEL = "gemma3:4b"
DEFAULT_LANGUAGE = "EN"

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
def call_ollama_llm(prompt, model):
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        return f"LLM error: {e.stderr.decode('utf-8')}"

# --- RAG Logic ---
def process_query(user_question, llm_model, lang):
    query_emb = embed_texts([user_question])[0]

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    if not docs or scores[0] < THRESHOLD_GOOD:
        fallback_prompt = f"{user_question}" if lang == "EN" else f"{user_question}"
        return call_ollama_llm(fallback_prompt, llm_model), []

    page_map = {}
    for doc, meta, score in zip(docs, metas, scores):
        url = meta.get("url", "")
        if score >= THRESHOLD_GOOD and url not in page_map:
            page_map[url] = {
                "text": doc,
                "score": round(score, 4)
            }
        elif score >= THRESHOLD_GOOD and url in page_map:
            page_map[url]["text"] += "\n" + doc

    page_contexts = [
        {"url": url, "text": data["text"], "score": data["score"]}
        for url, data in page_map.items()
    ]

    all_text = "\n\n".join(p["text"] for p in page_contexts)
    all_urls = [p["url"] for p in page_contexts]

    prompt = f"""
You are an expert assistant helping users understand technical documentation.

Sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation:
{all_text}

Question: {user_question}

Provide a detailed, structured, and practical answer. Include examples (e.g., Python, Bash) when applicable.
If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM.
"""
    return call_ollama_llm(prompt, llm_model), page_contexts

# --- Generate PDF ---
def generate_pdf(content):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        html_template = f"""
        <html>
        <head><meta charset='UTF-8'></head>
        <body><pre>{content}</pre></body>
        </html>
        """
        pisa.CreatePDF(html_template, dest=tmp_file)
        tmp_file.seek(0)
        pdf_data = tmp_file.read()
    return base64.b64encode(pdf_data).decode("utf-8")

# --- Dash App Setup ---
app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
app.title = "RAG Assistant"

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("🤖 RAG Assistant", className="text-primary"), width=8),
        dbc.Col(html.Img(src="https://www.svgrepo.com/show/331368/ai.svg", height="60px"), width=4, style={"textAlign": "right"})
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
                value=DEFAULT_LANGUAGE
            )
        ], width=4)
    ]),

    dbc.Row([
        dbc.Col(dbc.Button("Submit", id="submit-button", color="success"), width="auto"),
        dbc.Col(dbc.Button("🧹 Clear Output", id="clear-button", color="warning", className="ms-2"), width="auto"),
        dbc.Col(dbc.Checkbox(id="show-sources-toggle", value=False, className="ms-3"), width="auto"),
        dbc.Col(html.Label("Show sources", className="mt-2"), width="auto")
    ], className="my-3", align="center"),

    dbc.Card([html.Div(id="chat-history", children=[], style={"margin": "10px"})], color="dark", inverse=True),
    html.Div(id="pdf-download", className="mt-3 text-end"),
    dcc.Store(id="clear-question", data="")
], fluid=True, className="p-4")

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
    State("chat-history", "children"),
    prevent_initial_call=True
)
def update_chat(submit_clicks, clear_clicks, question, show_sources, llm_model, lang, history):
    triggered_id = ctx.triggered_id
    if triggered_id == "clear-button":
        return [], "", ""

    if not question:
        return history + [html.Div("❗ Please enter a question.")], "", question

    answer, source_data = process_query(question, llm_model, lang)
    formatted_answer = dcc.Markdown(answer)

    if show_sources and source_data:
        source_block = dcc.Markdown("\n".join([f"- {item['url']} (score: {item['score']})" for item in source_data]))
    else:
        source_block = ""

    pdf_content = f"You: {question}\n\nAnswer:\n{answer}\n\nSources:\n" + "\n".join([f"- {item['url']} (score: {item['score']})" for item in source_data])
    pdf_base64 = generate_pdf(pdf_content)
    download_link = html.A("📄 Download PDF", href=f"data:application/pdf;base64,{pdf_base64}", download="rag_answer.pdf", target="_blank", className="btn btn-outline-info")

    new_exchange = html.Div([
        html.H5("🧑 You:", className="text-warning"),
        html.Div(question, style={"whiteSpace": "pre-wrap", "marginBottom": "10px"}),
        html.H5("🤖 Assistant:", className="text-success"),
        formatted_answer,
        html.Div([html.Strong("Sources used:"), source_block], style={"marginTop": "10px", "color": "#aaa", "fontSize": "0.85em"})
    ], style={"marginBottom": "30px"})

    return history + [new_exchange], download_link, ""

if __name__ == "__main__":
    app.run(debug=True)
