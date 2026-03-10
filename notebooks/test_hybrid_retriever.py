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


query = "treatment for knee osteoarthritis"

results = hybrid.search(query, k=5)

print("\nQuery:", query)
print("\nHybrid Results:\n")

for r in results:
    print(r["text"][:300])
    print("------")