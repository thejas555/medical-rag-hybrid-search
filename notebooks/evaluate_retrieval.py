import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker


dense = DenseRetriever()
chunk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
dense.load_chunks(chunk_path)

sparse = SparseRetriever()
sparse.load_chunks(chunk_path)

hybrid = HybridRetriever(dense, sparse)

reranker = Reranker()


queries = [
    "treatment for knee osteoarthritis",
    "pain reduction in osteoarthritis trials",
    "biomarkers used in arthritis research",
    "inflammatory cytokines measured in arthritis",
    "clinical outcomes measured in knee osteoarthritis"
]


def precision_at_k(results, query):
    if not results:
        return 0.0

    stopwords = {"for", "in", "of", "and", "the", "to", "a", "an", "is", "are", "on", "with"}
    query_terms = [t for t in query.lower().split() if t not in stopwords]

    relevant = 0

    for r in results:

        text_words = set(re.findall(r'\w+', r["text"].lower()))

        matches = sum(1 for term in query_terms if term in text_words)

        if len(query_terms) >= 2:
            if matches >= 2:
                relevant += 1
        elif matches > 0:
            relevant += 1

    return relevant / len(results)


dense_scores = []
sparse_scores = []
hybrid_scores = []
rerank_scores = []

print("\nEvaluating Retrieval Systems\n")


for q in queries:

    print("\nQuery:", q)

    dense_results = dense.search(q, k=5)
    sparse_results = sparse.search(q, k=5)
    hybrid_results = hybrid.search(q, k=5)
    
    # For reranker, retrieve more (k=20) then rerank down to 5
    hybrid_results_20 = hybrid.search(q, k=20)
    reranked_results = reranker.rerank(q, hybrid_results_20, top_k=5)

    dense_p5 = precision_at_k(dense_results, q)
    sparse_p5 = precision_at_k(sparse_results, q)
    hybrid_p5 = precision_at_k(hybrid_results, q)
    rerank_p5 = precision_at_k(reranked_results, q)

    dense_scores.append(dense_p5)
    sparse_scores.append(sparse_p5)
    hybrid_scores.append(hybrid_p5)
    rerank_scores.append(rerank_p5)

    print("Dense Precision@5:", dense_p5)
    print("BM25 Precision@5:", sparse_p5)
    print("Hybrid Precision@5:", hybrid_p5)
    print("Reranked Precision@5:", rerank_p5)

    imp_dense = ((hybrid_p5 - dense_p5) / dense_p5) * 100 if dense_p5 > 0 else 0.0
    imp_sparse = ((hybrid_p5 - sparse_p5) / sparse_p5) * 100 if sparse_p5 > 0 else 0.0
    imp_rerank = ((rerank_p5 - hybrid_p5) / hybrid_p5) * 100 if hybrid_p5 > 0 else 0.0
    
    print(f"Hybrid Improvement: {imp_dense:+.1f}% vs Dense, {imp_sparse:+.1f}% vs BM25")
    print(f"Reranker Improvement: {imp_rerank:+.1f}% vs Hybrid")


dense_avg = sum(dense_scores) / len(dense_scores) if dense_scores else 0.0
sparse_avg = sum(sparse_scores) / len(sparse_scores) if sparse_scores else 0.0
hybrid_avg = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0.0
rerank_avg = sum(rerank_scores) / len(rerank_scores) if rerank_scores else 0.0

improvement_dense = ((hybrid_avg - dense_avg) / dense_avg) * 100 if dense_avg > 0 else 0.0
improvement_sparse = ((hybrid_avg - sparse_avg) / sparse_avg) * 100 if sparse_avg > 0 else 0.0
improvement_rerank = ((rerank_avg - hybrid_avg) / hybrid_avg) * 100 if hybrid_avg > 0 else 0.0

print("\n--- Final Overall Scores ---")
print(f"Average Dense Precision@5: {dense_avg:.3f}")
print(f"Average BM25 Precision@5: {sparse_avg:.3f}")
print(f"Average Hybrid Precision@5: {hybrid_avg:.3f}")
print(f"Average Reranked Precision@5: {rerank_avg:.3f}")
print(f"Overall Hybrid Improvement: {improvement_dense:+.1f}% vs Dense, {improvement_sparse:+.1f}% vs BM25")
print(f"Overall Reranker Improvement: {improvement_rerank:+.1f}% vs Hybrid")