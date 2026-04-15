import gzip
import json
import os

from sentence_transformers import SentenceTransformer
from vectorstore import VectorStore

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = []

gz_path = "nq_sample.jsonl.gz"
jsonl_path = "nq_sample.jsonl"

if os.path.exists(gz_path):
    file_path = gz_path
    open_fn = lambda path: gzip.open(path, "rt", encoding="utf-8")
elif os.path.exists(jsonl_path):
    file_path = jsonl_path
    open_fn = lambda path: open(path, "rt", encoding="utf-8")
else:
    raise FileNotFoundError(
        "Missing dataset file. Please add either 'nq_sample.jsonl.gz' or 'nq_sample.jsonl' to the project root."
    )

with open_fn(file_path) as f:
    for line in f:
        obj = json.loads(line)

        # ⚠️ pick real text field from dataset
        text = obj.get("text") or obj.get("document") or obj.get("question_text") or ""

        if text.strip():
            texts.append(text)

if not texts:
    raise ValueError(f"No text extracted from {file_path}")

# ✅ THIS is where encoding happens
embeddings = model.encode(texts, normalize_embeddings=True)

store = VectorStore(dim=embeddings.shape[1])
store.add(embeddings, texts)

store.save("rag_store.pkl")

print("Vector store built!")