## 📘 RAG Search Assistant – Centrale MEDITERRANEE Edition

**Author**: Anne-Laure MEALIER
**License**: GPL-3.0
**Version**: 1.3
**Last Updated**: 2024-05-14


### 🎯 Overview

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline, from **web crawling and enrichment** to **interactive question-answering** using a **local LLM** and **ChromaDB**.

It enables:

* Crawling technical documentation from any website
* Summarizing and enriching content using a local LLM via [Ollama](https://ollama.com/)
* Embedding text via SentenceTransformers (GPU supported)
* Storing embeddings in a **Chroma vector database**
* Querying this knowledge base through a **web interface** (Dash) using:

  * RAG only
  * LLM only
  * Hybrid (RAG + LLM)


## 📦 Project Structure

```bash
project-root/
├── assets/
│   └── logo_centrale.svg         # Centrale Médietrranée logo for the web app
├── logs/                         # Log files from pipeline and indexing
├── chroma_db/                    # Persistent ChromaDB store
├── enriched_pages.jsonl          # Output of RAG content enrichment
├── generateRAG.py                # Crawl + enrich + embed + store
├── vector_indexing.py           # Index JSONL to ChromaDB with weighted embeddings
├── embed_worker.py              # Fast GPU-ready embedding subprocess
├── searchEngineWebApp.py        # Dash app interface for querying
└── README.md                     # You are here 🚀
```


## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Example <code>requirements.txt</code></summary>

```txt
dash
dash-bootstrap-components
markdown
xhtml2pdf
beautifulsoup4
requests
chromadb
sentence-transformers
numpy
torch
scikit-learn
bs4
tiktoken
asyncio
```
 

</details>


### 2. Run the Full RAG Pipeline

```bash
python generateRAG.py https://your.website.com/
```

* Crawls all HTML pages from the base URL
* Summarizes each with an LLM
* Generates keyword lists
* Chunks content and stores enriched JSONL
* Computes vector embeddings
* Stores everything into ChromaDB


### 3. (Optional) Re-index with Weighted Embeddings

```bash
python vector_indexing.py enriched_pages.jsonl
```

* Uses a weighted combination of **summary** and **keywords**
* Optimized for fast GPU embedding


### 4. Launch the Dash Web App

```bash
python searchEngineWebApp.py
```

* Open `http://127.0.0.1:8050` in your browser
* Ask questions in English or French
* Toggle modes: Hybrid, RAG-only, or LLM-only
* Download responses as PDFs
* View sources with similarity scores


## 🧠 Architecture

### 1. **generateRAG.py**

* Crawls HTML pages from a base URL
* Extracts text, enriches with LLM (summary + keywords)
* Outputs a `.jsonl` file
* Embeds chunks and stores in ChromaDB

### 2. **vector\_indexing.py**

* (Optional) Reindexes `.jsonl` with custom weights:

  * `0.8 * summary + 0.2 * keywords`
* Uses `paraphrase-multilingual-MiniLM` with GPU acceleration

### 3. **embed\_worker.py**

* Lightweight subprocess for embedding queries
* Used by the Dash app
* Returns a vector from stdin JSON input

### 4. **searchEngineWebApp.py**

* Frontend with Dash and Bootstrap (CYBORG theme)
* User asks question → Query is embedded → Search ChromaDB
* LLM answers using RAG content or own knowledge
* PDF export and source display included


## 🖼️ Web Interface

* 🧠 Query interface with text area
* 🔁 Modes: RAG-only, Hybrid, LLM-only
* 📚 Show source URLs and scores (sorted by relevance)
* 📄 PDF export of full Q\&A session


## ⚙️ Configuration

Modify these constants in `searchEngineWebApp.py`:

```python
TOP_K = 50                # Number of top matches to retrieve
THRESHOLD_GOOD = 0.70     # Minimum score to consider a match relevant
DEFAULT_LLM_MODEL = "gemma3:4b"  # Ollama model to use
DEFAULT_QUERY_MODE = "rag_only"  # Starting mode
```


## 📌 Notes

* The system assumes Ollama is running locally at `http://localhost:11434`
* Embedding and inference prefer GPU (if available)
* All logs are stored in `/logs` with timestamps
* Files in `/assets` (like logos) are auto-served by Dash


## 🧪 Testing

To test locally:

```bash
# Crawl test site
python generateRAG.py https://doc.cc.in2p3.fr/

# Index generated file
python vector_indexing.py enriched_pages.jsonl

# Launch app
python searchEngineWebApp.py
```

## 📄 License

This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html)

You are free to use, modify, and distribute this software, provided that:

The source code remains open and publicly accessible under the same license.
Any derivative works or modified versions are also released under GPL-3.0.
Appropriate credit is given to the original author.


## 👩‍🔬 Author

**Anne-Laure MEALIER**
Centrale Méditerranée – 2024
Optimized for GPU acceleration and on-premise privacy
