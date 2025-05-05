# SearchEngine
A simple search engine using a RAG / LLM system on real data


### **Summary of the Web Crawling and Enrichment Pipeline**

This project consists of a full pipeline designed to crawl a website, extract its content, and enrich the data using an advanced LLM model (Ollama `deepseek-r1:14b`). The output is then indexed in a ChromaDB vector store to create a **Retrieval-Augmented Generation (RAG)** system.

#### **Key Components of the Pipeline:**

1. **Web Crawling:**

   * Crawls the provided base URL and its subpages (same domain).
   * Extracts the raw text content from each page.
   * Collects meta-information like page title and web path.

2. **Text Chunking:**

   * Divides the extracted text into manageable **paragraphs** or **chunks**.
   * Chunks are created to ensure they fit within a specified token limit for processing by the language model.

3. **Text Enrichment (via Ollama):**

   * For each chunk, the script calls **Ollama's deepseek-r1:14b** model to generate:

     * **Summary** (3-5 sentences)
     * **Keywords** (5-10 keywords related to the chunk)
   * This helps enhance the content by summarizing and extracting key concepts.

4. **Saving the Enriched Data:**

   * The enriched text (with summary and keywords) is saved to a **`.jsonl` file**.
   * Each entry includes:

     * `id`, `url`, `web_path`, `title`, `text` (chunked content), `summary`, `keywords`, and `timestamp`.

5. **ChromaDB Integration:**

   * The enriched data is indexed in **ChromaDB** for efficient **vector search**.
   * Chunks are stored as documents with metadata like title, keywords, summary, and web path, enabling powerful **retrieval-based generation (RAG)** in future queries.

6. **Main Functionality:**

   * The pipeline is wrapped in a `run_pipeline()` function, which handles the crawling, enrichment, and indexing processes.
   * A `main` method allows the user to easily invoke the pipeline with a given base URL.

---

### **End Result:**

By running this pipeline, you can:

* Crawl a website and extract its content.
* Enrich the content with summaries and keywords.
* Structure the data into `.jsonl` format.
* Build a **ChromaDB vector store** that allows fast search and retrieval of relevant content for AI-based applications (such as RAG).

