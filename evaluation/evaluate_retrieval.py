import os
import sys
import re

# Allow imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever


# ---------------------------
# 1. Load retrievers
# ---------------------------
chunk_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json')
)

dense = DenseRetriever()
dense.load_chunks(chunk_path)

sparse = SparseRetriever()
sparse.load_chunks(chunk_path)

hybrid = HybridRetriever(dense, sparse)


# ---------------------------
# 2. Evaluation queries
# ---------------------------
queries = [
    "treatment for knee osteoarthritis",
    "inflammatory cytokines in arthritis",
    "biomarkers in osteoarthritis",
    "pain reduction clinical trials osteoarthritis",
    "effects of prednisolone in arthritis",
    "role of IL-6 in inflammation",
    "clinical outcomes in knee osteoarthritis",
    "inflammatory markers measured in arthritis",
    "drug therapy for osteoarthritis",
    "methods used in arthritis trials"
]


# ---------------------------
# 3. Relevance function
# ---------------------------
def is_relevant(query, text):
    stopwords = {"for", "in", "of", "and", "the", "to", "a", "an", "is", "are", "on", "with"}
    query_terms = [t for t in query.lower().split() if t not in stopwords]
    
    # Extract words from text to prevent substring matching (e.g. "in" matching "insulin")
    text_words = set(re.findall(r'\w+', text.lower()))

    # Consider relevant if at least 2 meaningful terms match, or all if query is short
    matches = sum(1 for term in query_terms if term in text_words)
    if len(query_terms) >= 2:
        return matches >= 2
    return matches > 0


# ---------------------------
# 4. Precision@K
# ---------------------------
def precision_at_k(results, query, k=5):
    relevant = 0

    for r in results[:k]:
        if is_relevant(query, r["text"]):
            relevant += 1

    return relevant / k


# ---------------------------
# 5. Run evaluation
# ---------------------------
dense_scores = []
sparse_scores = []
hybrid_scores = []

print("\n--- Evaluating Retrieval Systems ---\n")

for query in queries:
    print(f"Query: {query}")

    dense_results = dense.search(query, k=5)
    sparse_results = sparse.search(query, k=5)
    hybrid_results = hybrid.search(query, k=5)

    dense_p5 = precision_at_k(dense_results, query)
    sparse_p5 = precision_at_k(sparse_results, query)
    hybrid_p5 = precision_at_k(hybrid_results, query)

    dense_scores.append(dense_p5)
    sparse_scores.append(sparse_p5)
    hybrid_scores.append(hybrid_p5)

    print(f"Dense Precision@5: {dense_p5:.2f}")
    print(f"Sparse Precision@5: {sparse_p5:.2f}")
    print(f"Hybrid Precision@5: {hybrid_p5:.2f}")
    
    imp_dense_pct = ((hybrid_p5 - dense_p5) / dense_p5) * 100 if dense_p5 > 0 else 0.0
    imp_sparse_pct = ((hybrid_p5 - sparse_p5) / sparse_p5) * 100 if sparse_p5 > 0 else 0.0
    print(f"Hybrid Improvement: {imp_dense_pct:+.1f}% vs Dense, {imp_sparse_pct:+.1f}% vs Sparse\n")


# ---------------------------
# 6. Final results
# ---------------------------
dense_avg = sum(dense_scores) / len(dense_scores) if dense_scores else 0.0
sparse_avg = sum(sparse_scores) / len(sparse_scores) if sparse_scores else 0.0
hybrid_avg = sum(hybrid_scores) / len(hybrid_scores) if hybrid_scores else 0.0

improvement_dense = ((hybrid_avg - dense_avg) / dense_avg) * 100 if dense_avg > 0 else 0.0
improvement_sparse = ((hybrid_avg - sparse_avg) / sparse_avg) * 100 if sparse_avg > 0 else 0.0

print("\n--- Final Results ---\n")
print(f"Average Dense Precision@5: {dense_avg:.3f}")
print(f"Average Sparse Precision@5: {sparse_avg:.3f}")
print(f"Average Hybrid Precision@5: {hybrid_avg:.3f}")
print(f"Overall Hybrid Improvement: {improvement_dense:+.2f}% vs Dense, {improvement_sparse:+.2f}% vs Sparse\n")