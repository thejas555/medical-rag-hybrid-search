import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):

        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)

        self.index = None
        self.chunks = None


    def load_chunks(self, chunk_path):

        with open(chunk_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        texts = [c["text"] for c in self.chunks]

        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)

        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        print("Building FAISS index...")
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        print("Index built with", self.index.ntotal, "vectors")


    def search(self, query, k=5):

        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, k)

        results = []

        for idx in indices[0]:
            results.append(self.chunks[idx])

        return results