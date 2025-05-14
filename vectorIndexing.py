"""
===============================================================================
 vector_indexing.py

 Author      : Anne-Laure MEALIER
 Created     : 2024-05-14
 License     : GPL-3.0
 Version     : 1.0

 Description :
 ------------------------------------------------------------------------------
 This script loads a JSONL file containing enriched and chunked web content
 (produced by a separate data pipeline), and indexes each chunk into a ChromaDB
 vector store using sentence embeddings.

 It is useful when the crawl and enrichment steps have already been completed,
 and you simply want to create or refresh the vector index without reprocessing
 the website content.

 Features :
 - Reads pre-enriched documents from a .jsonl file
 - Uses sentence-transformers to compute vector embeddings
 - Stores documents and metadata into a persistent ChromaDB collection
 - Logs progress, document count, and timing to both console and log file

 Usage :
 -------
 Run directly from the command line:
     python vector_indexing.py

 Make sure `enriched_pages.jsonl` is present in the working directory,
 and that the `chromadb` database folder exists or will be created.

 Dependencies :
  - chromadb
  - sentence-transformers
  - numpy
  - json
  - logging
===============================================================================
"""

import os
import json
import time
import logging
from datetime import datetime
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

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


def index_vector_store(
    jsonl_input_path: str,
    chroma_collection_name: str = "web_chunks",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 4000
):
    """
    Loads enriched chunks from JSONL and indexes them into ChromaDB in batches
    """
    logger.info("🧠 Starting vector indexing...")

    index_start = time.time()

    client = chromadb.Client(Settings(persist_directory=persist_directory))
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
    )

    documents, metadatas, ids = [], [], []
    try:
        with open(jsonl_input_path, 'r', encoding='utf-8') as f:
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

        total = len(documents)
        logger.info(f"📦 Loaded {total} documents from {jsonl_input_path}")

        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            collection.add(documents=batch_docs, metadatas=batch_meta, ids=batch_ids)
            logger.info(f"✅ Indexed batch {i // batch_size + 1} — {len(batch_docs)} documents")

        index_end = time.time()
        logger.info(f"✅ Vector store built in {index_end - index_start:.2f} seconds")
        logger.info(f"📚 Total documents indexed: {total}")

    except Exception as e:
        logger.error(f"❌ Failed to index vector store: {e}")


if __name__ == "__main__":
    import sys
    start_time = time.time()
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "enriched_pages.jsonl"
    index_vector_store(jsonl_input_path=jsonl_path)
    total = time.time() - start_time
    logger.info(f"🎉 Indexing process completed in {total:.2f} seconds")
