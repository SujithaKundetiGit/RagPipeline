import faiss
import numpy as np
import pickle

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vec, k=3):
        distances, indices = self.index.search(
            np.array([query_vec]).astype("float32"), k
        )

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                "text": self.texts[idx],
                "score": float(distances[0][i])
            })
        return results

    # ✅ SAVE PIPELINE
    def save(self, path="vectorstore.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "index": faiss.serialize_index(self.index),
                "texts": self.texts
            }, f)

    # ✅ LOAD PIPELINE
    @staticmethod
    def load(path="vectorstore.pkl"):
        import faiss
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = VectorStore(dim=384)
        obj.index = faiss.deserialize_index(data["index"])
        obj.texts = data["texts"]
        return obj