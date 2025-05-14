# 🔍 Conversational Search over Documentation

This project provides a full pipeline and Streamlit interface for building a **RAG (Retrieval-Augmented Generation)** system over technical documentation hosted at [https://doc.cc.in2p3.fr/](https://doc.cc.in2p3.fr/). It scrapes the site, builds semantic vector embeddings, and allows users to ask questions and receive LLM-generated answers based on retrieved relevant content.


## 📦 Project Structure

```
project/
│
├── pipeline.py            # Scrapes the site, chunks text, generates summaries/keywords, builds Chroma vector store
├── app.py                 # Streamlit interface for querying the vector store and getting LLM responses
├── chroma_db/             # Persisted ChromaDB collection
├── enriched_pages.jsonl   # Enriched and chunked text content (optional intermediate file)
├── README.md              # This documentation
```

## 🧠 Technologies Used

| Component     | Tech/Model                                                       |
| ------------- | ---------------------------------------------------------------- |
| **Scraper**   | `requests`, `BeautifulSoup`                                      |
| **Embedding** | `sentence-transformers/all-mpnet-base-v2`                        |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/)                           |
| **LLM**       | [Ollama](https://ollama.com) with `mistral:7b`, `deepseek`, etc. |
| **Frontend**  | [Streamlit](https://streamlit.io/)                               |



## ⚙️ How It Works

### Step 1: Scraping and Indexing

The `pipeline.py` script:

* Crawls all HTML pages starting from a base URL.
* Extracts text, chunks it by paragraph.
* Calls a local LLM (via Ollama) to generate:

  * A short summary (3–5 sentences)
  * A list of keywords (5–10)
* Saves everything to a JSONL file.
* Encodes each chunk into vectors using a SentenceTransformer.
* Stores vectors and metadata into a **ChromaDB** persistent collection.

### Step 2: Search and Chat Interface

The `app.py` Streamlit app:

* Takes a natural language question from the user.
* Encodes it into a query embedding.
* Retrieves the top-k relevant chunks from the vector store using a **weighted embedding**:

```
combined_embedding = 0.7 * summary_embedding + 0.3 * keyword_embedding
```

* Based on similarity score:

  * ✅ **High (≥ 70%)**: Answer with LLM + context.
  * ⚠️ **Medium (40–69%)**: Show results and let user trigger LLM.
  * 🔴 **Low (< 40%)**: Generate answer using LLM only, no context.


## 🚀 Quick Start

### 1. Run the data pipeline

```bash
python pipeline.py
```

This will create and store the vector collection in `./chroma_db`.

### 2. Start the app

```bash
streamlit run app.py
```

Make sure the Ollama server is running locally (`ollama serve`) and the models are available (e.g., `mistral`, `deepseek`, etc.).


## 🧪 Example Query Flow

1. User inputs: **"How do I configure a Python environment?"**
2. The system finds top 5 relevant chunks (summaries + keywords).
3. If confidence is high, LLM answers using these summaries.
4. Otherwise, user can choose to invoke LLM with or without context.


## 📊 Similarity Logic

* Embeddings are generated for both:

  * **Summary** of a chunk
  * **Keywords** extracted from the same chunk

* These are combined:

  ```python
  0.7 * summary_embedding + 0.3 * keyword_embedding
  ```

* Cosine similarity is computed with the query embedding.


## 🔧 Configuration Options

You can adjust these in `app.py`:

```python
SUMMARY_WEIGHT = 0.7        # Controls weight of summary in similarity calc
KEYWORDS_WEIGHT = 0.3       # Controls weight of keywords
THRESHOLD_GOOD = 0.70       # Above this, use LLM confidently
THRESHOLD_LOW = 0.40        # Below this, fallback to LLM without context
```

## 📁 Requirements

Make sure you install:

```bash
pip install streamlit sentence-transformers chromadb scikit-learn beautifulsoup4 requests bs4
```

Also install and run [Ollama](https://ollama.com/) locally for LLM support.


## 🛠️ TODO / Next Steps

* [ ] Add ability to tweak weights in UI.
* [ ] Visualize similarity scores as progress bars.
* [ ] Export conversation or results.
* [ ] Add PDF/document support.


## 📝 License

MIT License. Open for academic and non-commercial usage.
