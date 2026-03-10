import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sparse_retriever import SparseRetriever


retriever = SparseRetriever()

chunk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
retriever.load_chunks(chunk_path)


query = "What provides a way to quantify corticomotor drive during a functional task?"

results = retriever.search(query, k=5)

print("\nQuery:", query)

print("\nTop BM25 results:\n")

for r in results:
    print(r["text"][:300])
    print("------")