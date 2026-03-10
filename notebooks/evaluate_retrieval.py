import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever


dense = DenseRetriever()
chunk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
dense.load_chunks(chunk_path)

sparse = SparseRetriever()
sparse.load_chunks(chunk_path)

hybrid = HybridRetriever(dense, sparse)


queries = [
    "treatment for knee osteoarthritis",
    "pain reduction in osteoarthritis trials",
    "biomarkers used in arthritis research",
    "inflammatory cytokines measured in arthritis",
    "clinical outcomes measured in knee osteoarthritis"
]


def precision_at_k(results, query):

    query_terms = query.lower().split()

    relevant = 0

    for r in results:

        text = r["text"].lower()

        matches = sum(1 for term in query_terms if term in text)

        if matches >= 2:
            relevant += 1

    return relevant / len(results)


print("\nEvaluating Retrieval Systems\n")


for q in queries:

    print("\nQuery:", q)

    dense_results = dense.search(q, k=5)
    sparse_results = sparse.search(q, k=5)
    hybrid_results = hybrid.search(q, k=5)

    print("Dense Precision@5:", precision_at_k(dense_results, q))
    print("BM25 Precision@5:", precision_at_k(sparse_results, q))
    print("Hybrid Precision@5:", precision_at_k(hybrid_results, q))