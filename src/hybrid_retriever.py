import numpy as np


class HybridRetriever:

    def __init__(self, dense_retriever, sparse_retriever, alpha=0.6):

        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.alpha = alpha


    def normalize(self, scores):

        scores = np.array(scores)

        if scores.max() == scores.min():
            return scores

        return (scores - scores.min()) / (scores.max() - scores.min())


    def search(self, query, k=5):

        # Dense search
        query_embedding = self.dense.model.encode([query])
        distances, indices = self.dense.index.search(query_embedding, 50)

        dense_scores = -distances[0]

        dense_norm = self.normalize(dense_scores)

        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.sparse.bm25.get_scores(tokenized_query)

        bm25_norm = self.normalize(bm25_scores)

        score_dict = {}

        for i, idx in enumerate(indices[0]):

            chunk = self.dense.chunks[idx]

            dense_score = dense_scores[i]
            dense_n = dense_norm[i]

            bm25_score = bm25_scores[idx]
            bm25_n = bm25_norm[idx]

            final_score = self.alpha * dense_n + (1 - self.alpha) * bm25_n

            score_dict[idx] = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "dense_score": float(dense_score),
                "bm25_score": float(bm25_score),
                "dense_norm": float(dense_n),
                "bm25_norm": float(bm25_n),
                "final_score": float(final_score)
            }

        ranked = sorted(score_dict.values(), key=lambda x: x["final_score"], reverse=True)

        return ranked[:k]