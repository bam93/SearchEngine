# 🔍 Conversational Search Engine for Documentation

This project implements a full **Retrieval-Augmented Generation (RAG)** pipeline using Python, designed to crawl technical documentation, embed and index its content, and provide a conversational interface for querying the knowledge base using a large language model (LLM).



## 📁 Project Structure

```bash
.
├── ragPipelineOllama.py       # Script to crawl, enrich, and index web content
├── searchEngineAPI.py         # Streamlit-based search and chat UI
├── logs/                      # Automatically created log files of pipeline execution
├── chroma_db/                 # Vector store (created at runtime)
├── enriched_pages.jsonl       # JSONL file containing enriched chunks (created at runtime)
```

## 🚀 Features

* ✅ Web crawler with HTML content filtering
* ✅ Chunking of text content into manageable paragraphs
* ✅ Summary and keyword extraction using a local LLM via Ollama
* ✅ Storage in [ChromaDB](https://www.trychroma.com/) vector database
* ✅ Query interface with document similarity scoring
* ✅ Conversational LLM responses with fallback strategies
* ✅ Real-time execution progress and ETA estimation
* ✅ Logging to both console and `logs/` folder

## 🧠 Technologies Used

| Component  | Tech Stack / Tools                              |
| ---------- | ----------------------------------------------- |
| Crawler    | `requests`, `BeautifulSoup`                     |
| Chunking   | Paragraph-based with token limits               |
| Enrichment | `Ollama` (e.g. `mistral:7b`, `deepseek`)        |
| Embeddings | `sentence-transformers`, `ChromaDB`             |
| Interface  | `Streamlit`                                     |
| Similarity | Weighted cosine similarity (summary + keywords) |
| Logging    | Python `logging`, auto-named `.log` files       |



## 🧪 How It Works

### 1. `ragPipelineOllama.py`: Build the Vector Store

This script performs the following:

* Crawls a given base URL (e.g., [https://doc.cc.in2p3.fr/](https://doc.cc.in2p3.fr/))
* Skips irrelevant file types (images, binaries, archives, etc.)
* Extracts main content and chunks it into paragraphs
* Sends each page to a local LLM via Ollama to get:

  * ✅ A short summary
  * ✅ A list of keywords
* Saves each chunk into a `.jsonl` file
* Creates embeddings (summary + keywords) and stores everything in ChromaDB

You can monitor progress via:

* Console and log output (page/chunk counts, ETA)
* Automatically generated logs under `logs/`

### 2. `searchEngineAPI.py`: Ask Questions via UI

This script runs a Streamlit app that allows users to:

* Input a natural language question
* Find the top-5 most relevant document chunks (based on weighted summary/keyword embeddings)
* View document info (summary, keywords, source URL)
* Get LLM-generated answers depending on similarity level:

  * 🔵 High (≥ 70%): direct LLM answer using context
  * 🟡 Medium (40–69%): suggestion to ask LLM
  * 🔴 Low (< 40%): LLM response without context


## 📦 Setup Instructions

### Requirements

Install dependencies:

```bash
pip install streamlit sentence-transformers chromadb scikit-learn beautifulsoup4 requests bs4 sentence_transformers tiktoken dash torch asyncio
```

Dependencies include:

* `requests`
* `beautifulsoup4`
* `chromadb`
* `sentence-transformers`
* `scikit-learn`
* `streamlit`
* `numpy`

### Start Ollama Server

Make sure Ollama is running locally with your preferred model:

```bash
ollama serve
ollama run deepseek-r1:14b
```


## 🧰 Usage

### Step 1: Run the Data Pipeline

```bash
python ragPipelineOllama.py
```

This will generate:

* `enriched_pages.jsonl`
* Populate your local ChromaDB vector store

### Step 2: Launch the Streamlit UI

```bash
streamlit run searchEngineAPI.py
```

Navigate to `http://localhost:8501` in your browser and start asking questions!


## 📊 Logs and Monitoring

* Execution logs are written both to console and to timestamped files under `./logs/`
* Example: `logs/pipeline_log_20240514_1830.log`
* These logs include:

  * Pages processed
  * Chunks generated
  * ETA estimation
  * Errors (if any)


## 📌 Example Use Case

> Ask: **"How do I install a Python package with Conda?"**

You’ll see:

* The top matching document(s)
* Their summary and keywords
* An AI-generated response using context, if available


## 🔒 Limitations & Considerations

* Only HTML pages are processed — binary formats (PDFs, images, etc.) are skipped
* LLM must be running locally via [Ollama](https://ollama.com/)
* Large crawls may consume considerable time and memory


## 📁 Optional Improvements

* Add PDF parsing support with `PyMuPDF` or `pdfminer`
* Use language-specific chunking (e.g., sentence-based for multi-lingual support)
* Add vector search filtering by topic, date, or section
* Support multiple base URLs or domains


## 📝 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

You are free to use, modify, and distribute this software, provided that:
- The source code remains open and publicly accessible under the same license.
- Any derivative works or modified versions are also released under GPL-3.0.
- Appropriate credit is given to the original author.

For full terms, see the [GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html).
