# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# File        : generateRAG.py
# Description : Crawl and enrich web documentation using LLM, store in ChromaDB
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.1 (GPU-optimized)
#
# This script performs a complete RAG (Retrieval-Augmented Generation) pipeline:
# - Crawls HTML pages from a specified base URL
# - Extracts readable content and chunks it into paragraphs
# - Sends each page to a local LLM via Ollama to generate:
#     • a summary (3–5 sentences)
#     • a list of keywords
# - Saves the enriched data to a .jsonl file
# - Converts the enriched content into vector embeddings using GPU if available
# - Stores documents and metadata in a persistent ChromaDB vector store
# -----------------------------------------------------------------------------

import os
import re
import time
import json
import uuid
import torch
import logging
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"pipeline_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Device Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🖥️ Using device: {device}")

# === Embedding Model ===
embedding_model = "sentence-transformers/all-mpnet-base-v2"
embedder = SentenceTransformer(embedding_model, device=device)

def run_pipeline(
    base_url: str,
    jsonl_output_path: str = "enriched_pages.jsonl",
    chroma_collection_name: str = "web_chunks",
    model: str = "deepseek-r1:14b",
    batch_size: int = 4000
):
    logger.info("🚀 Starting RAG pipeline")

    def ollama_generate(prompt, model=model):
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False}
        )
        return response.json()['response']

    def extract_summary_and_keywords(text):
        prompt = f"""
Here is a web article:

{text[:1500]}

Please return:
1. A short summary (3-5 sentences)
2. A list of 5 to 10 keywords

Expected JSON format:
{{
  "summary": "...",
  "keywords": ["word1", "word2", ...]
}}
"""
        try:
            raw = ollama_generate(prompt)
            json_part = raw[raw.find('{'):raw.rfind('}')+1]
            return json.loads(json_part)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"summary": "", "keywords": []}

    def split_into_paragraphs(text, max_len=512):
        paras = [p.strip() for p in text.split('\n') if p.strip()]
        chunks, current = [], ''
        for para in paras:
            if len(current) + len(para) < max_len:
                current += ' ' + para
            else:
                chunks.append(current.strip())
                current = para
        if current:
            chunks.append(current.strip())
        return chunks

    visited, to_visit = set(), set([base_url])
    page_counter = 0
    chunk_counter = 0

    with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
        while to_visit:
            url = to_visit.pop()
            if url in visited:
                continue

            try:
                response = requests.get(url)
                content_type = response.headers.get('Content-Type', '')
                if not content_type.startswith("text/html"):
                    continue

                visited.add(url)

                if response.status_code == 200:
                    page_counter += 1
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    title = soup.title.string.strip() if soup.title else ''
                    web_path = urlparse(url).path
                    enrichments = extract_summary_and_keywords(text)

                    chunks = split_into_paragraphs(text)
                    for idx, chunk in enumerate(chunks):
                        chunk_counter += 1
                        doc = {
                            "id": str(uuid.uuid4()),
                            "url": url,
                            "web_path": web_path,
                            "title": title,
                            "text": chunk,
                            "summary": enrichments.get("summary", ""),
                            "keywords": enrichments.get("keywords", []),
                            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                            "chunk_id": idx
                        }
                        f_out.write(json.dumps(doc, ensure_ascii=False) + '\n')

                    excluded_extensions = re.compile(
                        r".*\.(png|jpe?g|gif|bmp|svg|webp|pdf|zip|tar|gz|tar\\.gz|rar|7z"
                        r"|docx?|xlsx?|pptx?|exe|msi|sh|bin|iso|dmg|apk|jar"
                        r"|mp3|mp4|avi|mov|ogg|wav"
                        r"|ttf|woff2?|eot"
                        r"|ics|csv|dat)(\?.*)?$", re.IGNORECASE
                    )

                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])
                        if (
                            full_url.startswith(base_url)
                            and full_url not in visited
                            and not excluded_extensions.match(full_url)
                        ):
                            to_visit.add(full_url)

                time.sleep(1)

            except Exception as e:
                logger.error(f"❌ Error with {url}: {e}")

    logger.info(f"🌐 Finished crawling {page_counter} pages | {chunk_counter} chunks")

    logger.info("🧠 Starting vector indexing...")
    client = PersistentClient(path='./chroma_db')
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    documents, metadatas, ids = [], [], []
    with open(jsonl_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            documents.append(entry['text'])
            metadatas.append({
                "title": entry['title'],
                "url": entry['url'],
                "keywords": ", ".join(entry["keywords"]) if isinstance(entry["keywords"], list) else entry["keywords"],
                "summary": entry['summary'],
                "web_path": entry['web_path']
            })
            ids.append(entry['id'])

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_meta = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
        logger.info(f"✅ Indexed batch {i // batch_size + 1} — {len(batch_docs)} documents")

    logger.info(f"✅ Total documents indexed: {len(documents)}")
    logger.info("🎉 Pipeline completed.")


if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://doc.cc.in2p3.fr/"
    run_pipeline(base_url=url)
