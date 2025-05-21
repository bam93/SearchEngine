# -----------------------------------------------------------------------------
# Author      : Anne-Laure MEALIER
# Organization: Centrale Méditerranée
# File        : embed_worker.py
# Description : Lightweight subprocess embedding script using SentenceTransformers.
# Created     : 2024-05-15
# License     : GPL-3.0
# Version     : 1.0
#
# This script reads a JSON-encoded list of texts from stdin, computes embeddings
# using a multilingual SentenceTransformer model (GPU-accelerated if available),
# and outputs the embeddings as a JSON list to stdout.
#
# Intended for use as a subprocess utility inside the RAG assistant pipeline.
# -----------------------------------------------------------------------------


import sys
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=device)

    # Read stdin
    raw = sys.stdin.read()
    texts = json.loads(raw)

    if not isinstance(texts, list):
        raise ValueError("Input must be a list of strings.")

    # Compute embeddings
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)

    # Output as JSON
    json.dump(embeddings.tolist(), sys.stdout)

except Exception as e:
    print(f"❌ embed_worker.py error: {str(e)}", file=sys.stderr)
    sys.exit(1)
