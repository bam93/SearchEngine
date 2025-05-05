import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import uuid
import time
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def run_pipeline(
    base_url: str,
    jsonl_output_path: str = "enriched_pages.jsonl",
    chroma_collection_name: str = "web_chunks",
    model: str = "deepseek-r1:14b"
):
    """
    Full pipeline to:
    - Crawl a subsite
    - Extract and chunk text
    - Enrich each chunk with summary and keywords using Ollama
    - Save enriched chunks to a JSONL file
    - Index documents into ChromaDB vector store
    """

    # === Ollama Call ===
    def ollama_generate(prompt, model=model):
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={'model': model, 'prompt': prompt, 'stream': False}
        )
        return response.json()['response']

    # === Extract summary + keywords from text ===
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
            print(f"LLM error: {e}")
            return {"summary": "", "keywords": []}

    # === Paragraph-based text chunking ===
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

    # === Crawl subsite and save JSONL ===
    visited, to_visit = set(), set([base_url])
    with open(jsonl_output_path, 'w', encoding='utf-8') as f_out:
        while to_visit:
            url = to_visit.pop()
            if url in visited:
                continue

            try:
                response = requests.get(url)
                visited.add(url)

                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                    title = soup.title.string.strip() if soup.title else ''
                    web_path = urlparse(url).path
                    enrichments = extract_summary_and_keywords(text)

                    # Process and save each paragraph chunk
                    for idx, chunk in enumerate(split_into_paragraphs(text)):
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

                    # Collect all links to crawl within the same domain
                    for link in soup.find_all('a', href=True):
                        full_url = urljoin(url, link['href'])
                        if full_url.startswith(base_url) and full_url not in visited:
                            to_visit.add(full_url)

                time.sleep(1)  # avoid hammering the server

            except Exception as e:
                print(f"Error with {url}: {e}")

    # === Create ChromaDB vector store ===
    client = chromadb.Client(Settings(persist_directory='./chroma_db'))
    collection = client.get_or_create_collection(
        name=chroma_collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="google/bigbird-roberta-large"  
            # altaidevorg/bge-m3-distill-8l
            # sentence-transformers/all-mpnet-base-v2
            # google/bigbird-roberta-large
            # OrdalieTech/Solon-embeddings-large-0.1   
        )
    )

    # Read the enriched JSONL and load into vector store
    with open(jsonl_output_path, 'r', encoding='utf-8') as f:
        documents, metadatas, ids = [], [], []
        for line in f:
            entry = json.loads(line)
            documents.append(entry['text'])
            metadatas.append({
                "title": entry['title'],
                "url": entry['url'],
                "keywords": entry['keywords'],
                "summary": entry['summary'],
                "web_path": entry['web_path']
            })
            ids.append(entry['id'])

        collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print("Pipeline completed: Crawled, enriched, chunked, saved and indexed.")


if __name__ == "__main__":
    # Example usage
    run_pipeline(base_url="https://doc.cc.in2p3.fr/")