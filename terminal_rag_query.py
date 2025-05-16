# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# Organization: Centrale Méditerranée
# File        : terminal_rag_query.py
# Description : Interactive terminal-based RAG assistant using ChromaDB and local LLMs.
# Created     : 2024-05-15
# License     : GPL-3.0
# Version     : 1.1 (Langue dynamique + seuil stricte)
# -----------------------------------------------------------------------------

import json
import subprocess
import numpy as np
import requests
import tempfile
from chromadb import PersistentClient

TOP_K = 50
THRESHOLD_GOOD = 0.70
DEFAULT_LLM_MODEL = "gemma3:4b"
DEFAULT_LANGUAGE = "EN"
DEFAULT_QUERY_MODE = "rag_only"

client = PersistentClient(path="./chroma_db")
collection = client.get_collection(name="web_chunks")

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

def call_ollama_llm(prompt, model, temperature=0.1):
    try:
        payload = {"model": model, "prompt": prompt, "options": {"temperature": temperature}}
        response = requests.post("http://localhost:11434/api/generate", json=payload, stream=True)
        if response.status_code == 200:
            return ''.join(json.loads(line.decode("utf-8"))["response"]
                           for line in response.iter_lines() if line and "response" in json.loads(line.decode("utf-8")))
        else:
            return f"LLM API error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"LLM API exception: {str(e)}"

def get_translations(lang):
    return {
        "intro": "Vous êtes un assistant expert qui aide les utilisateurs à comprendre de la documentation technique." if lang == "FR"
                 else "You are an expert assistant helping users understand technical documentation.",
        "instr_rag": "Fournissez une réponse détaillée, structurée et pratique uniquement à partir de la documentation fournie." if lang == "FR"
                     else "Provide a detailed, structured, and practical answer using only the provided documentation.",
        "instr_hybrid": "Ajoutez des compléments issus du LLM si pertinent, en les identifiant clairement." if lang == "FR"
                        else "If relevant, enhance the response with complementary LLM knowledge and clearly indicate what part comes from the LLM.",
        "no_docs": "⚠️ Aucun document pertinent trouvé." if lang == "FR"
                   else "⚠️ No relevant documents found."
    }

def process_query(user_question, llm_model, lang, mode=DEFAULT_QUERY_MODE):
    temperature = 0.1 if mode == "rag_only" else 0.4 if mode == "hybrid" else 0.7
    t = get_translations(lang)

    if mode == "llm_only":
        return call_ollama_llm(user_question, llm_model, temperature), []

    query_emb = embed_texts([user_question])[0]
    results = collection.query(query_embeddings=[query_emb], n_results=TOP_K, include=["documents", "metadatas", "distances"])

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    scores = results.get("distances", [[]])[0]

    relevant = [(doc, meta, score) for doc, meta, score in zip(docs, metas, scores) if score >= THRESHOLD_GOOD]

    if not relevant:
        return t["no_docs"], []

    page_map = {}
    for doc, meta, score in relevant:
        url = meta.get("url", "")
        page_map.setdefault(url, {"text": "", "score": round(score, 4)})
        page_map[url]["text"] += "\n" + doc

    page_contexts = [{"url": url, "text": d["text"], "score": d["score"]} for url, d in page_map.items()]
    all_text = "\n\n".join(p["text"] for p in page_contexts)
    all_urls = [p["url"] for p in page_contexts]

    prompt = f"""{t['intro']}

Sources:
{chr(10).join('- ' + url for url in all_urls)}

Documentation:
{all_text}

Question: {user_question}

{t['instr_rag']}
"""
    if mode == "hybrid":
        prompt += f"\n{t['instr_hybrid']}"

    return call_ollama_llm(prompt, llm_model, temperature), page_contexts

def generate_pdf(content):
    from xhtml2pdf import pisa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        html_template = f"<html><body><pre>{content}</pre></body></html>"
        pisa.CreatePDF(html_template, dest=tmp_file)
        tmp_file.seek(0)
        return tmp_file.name

def main():
    print("📚 Welcome to RAG Terminal Assistant")
    llm_model = DEFAULT_LLM_MODEL
    mode = DEFAULT_QUERY_MODE
    lang = DEFAULT_LANGUAGE

    while True:
        print("\n" + "-" * 40)
        question = input("🧑 Your question (or 'exit'): ").strip()
        if question.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        print(f"\n⏳ Mode: {mode.upper()} | Langue: {lang} — Please wait...")
        answer, sources = process_query(question, llm_model, lang, mode)

        print("\n🤖 Assistant:\n", answer)

        if sources:
            print("\n🔗 Sources:")
            for s in sorted(sources, key=lambda x: x["score"], reverse=True):
                print(f"- {s['url']} (score: {s['score']})")

        if input("\n💾 Export to PDF? (y/n): ").strip().lower() == "y":
            content = f"You: {question}\n\nAnswer:\n{answer}\n\nSources:\n" + \
                      "\n".join([f"- {s['url']} (score: {s['score']})" for s in sources])
            path = generate_pdf(content)
            print(f"✅ PDF saved to: {path}")

        if input("\n⚙️ Change mode/lang? (y/n): ").strip().lower() == "y":
            mode_input = input("Mode (rag_only / hybrid / llm_only): ").strip().lower()
            if mode_input in ["rag_only", "hybrid", "llm_only"]:
                mode = mode_input
            lang_input = input("Language (EN / FR): ").strip().upper()
            if lang_input in ["EN", "FR"]:
                lang = lang_input

if __name__ == "__main__":
    main()
