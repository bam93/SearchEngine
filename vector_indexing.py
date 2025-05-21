# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# Organization: Centrale Méditerranée
# File        : vector_indexing.py
# Description : Indexes enriched JSONL into ChromaDB with GPU-accelerated embeddings
# Created     : 2024-05-14
# License     : GPL-3.0
# Version     : 1.4 (Optimized for batching and embedding performance)
# -----------------------------------------------------------------------------

import os
import json
import time
import logging
from datetime import datetime
from chromadb import PersistentClient
from chromadb.utils.embedding_functions import EmbeddingFunction

# === Logger Setup ===
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"indexing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename, encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# === Embedding Configuration ===
SUMMARY_WEIGHT = 0.8
KEYWORDS_WEIGHT = 0.2

# === Custom Fast GPU Embedder ===
class GPUEmbedder(EmbeddingFunction):
    def name(self):
        return "sentence_transformer"
    def __init__(self):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=self.device)
        logger.info(f"✅ Embedding model loaded on: {self.device}")

    def __call__(self, texts):
        import numpy as np
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        if np.all(embeddings == 0):
            logger.warning("⚠️ All generated embeddings are zeros.")
        return embeddings

# === Generate Weighted Embeddings in Batch ===
def generate_weighted_embeddings(embedder, summaries, keywords_list, log_samples=5):
    import numpy as np
    logger.info("⚙️ Encoding summaries in batch...")
    summary_embs = embedder(summaries)
    logger.info("⚙️ Encoding keywords in batch...")
    keyword_embs = embedder(keywords_list)

    weighted_embeddings = []
    for idx, (s_emb, k_emb) in enumerate(zip(summary_embs, keyword_embs)):
        weighted_emb = SUMMARY_WEIGHT * s_emb + KEYWORDS_WEIGHT * k_emb
        weighted_embeddings.append(weighted_emb.tolist())
        if idx < log_samples:
            norm = np.linalg.norm(weighted_emb)
            logger.info(f"🔍 Norm of embedding for sample {idx + 1}: {norm:.4f}")
    return weighted_embeddings

# === Indexer Function ===
def index_vector_store(
    jsonl_input_path: str,
    chroma_collection_name: str = "web_chunks",
    persist_directory: str = "./chroma_db",
    batch_size: int = 4000
):
    logger.info("🧠 Starting vector indexing...")
    index_start = time.time()

    embedder = GPUEmbedder()
    client = PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=None  # We supply embeddings manually
    )

    documents, metadatas, ids = [], [], []
    summaries, keywords_list = [], []

    try:
        with open(jsonl_input_path, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                summary = entry.get("summary", "")
                keywords = ", ".join(entry["keywords"]) if isinstance(entry["keywords"], list) else entry["keywords"]

                summaries.append(summary)
                keywords_list.append(keywords)

                documents.append(entry['text'])
                metadatas.append({
                    "title": entry['title'],
                    "url": entry['url'],
                    "keywords": keywords,
                    "summary": summary,
                    "web_path": entry['web_path']
                })
                ids.append(entry['id'])

        total = len(documents)
        logger.info(f"📦 Loaded {total} documents from {jsonl_input_path}")

        embeddings = generate_weighted_embeddings(embedder, summaries, keywords_list)

        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embs = embeddings[i:i + batch_size]
            collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids, embeddings=batch_embs)
            logger.info(f"✅ Indexed batch {i // batch_size + 1} — {len(batch_docs)} documents")

        index_end = time.time()
        logger.info(f"✅ Vector store built in {index_end - index_start:.2f} seconds")
        logger.info(f"📚 Total documents indexed: {total}")

        if not os.path.isdir(persist_directory):
            logger.warning(f"⚠️ Warning: Expected directory '{persist_directory}' was not created.")
        else:
            logger.info(f"💾 Vector store persisted under: {persist_directory}")

    except Exception as e:
        logger.error(f"❌ Failed to index vector store: {e}")

    logger.info("💾 Persisted vector store to disk.")

# === Entry Point ===
if __name__ == "__main__":
    import sys
    start_time = time.time()
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_pages.jsonl"
    index_vector_store(jsonl_input_path=jsonl_path)
    total = time.time() - start_time
    logger.info(f"🎉 Indexing process completed in {total:.2f} seconds")
