import json
from rank_bm25 import BM25Okapi


class SparseRetriever:

    def __init__(self):
        self.bm25 = None
        self.chunks = None
        self.tokenized_corpus = None

    def load_chunks(self, chunk_path):

        with open(chunk_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        corpus = [chunk["text"] for chunk in self.chunks]

        self.tokenized_corpus = [doc.lower().split() for doc in corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print("BM25 index built with", len(self.tokenized_corpus), "documents")

    def search(self, query, k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        results = [self.chunks[i] for i in top_indices]

        return results
