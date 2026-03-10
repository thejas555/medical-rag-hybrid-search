import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dense_retriever import DenseRetriever


retriever = DenseRetriever()

chunk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
retriever.load_chunks(chunk_path)


query = "effectiveness of continuous wound infusion of ropivacaine in postoperative pain relief"

results = retriever.search(query, k=5)

print("\nQuery:", query)

print("\nTop results:\n")

for r in results:
    print(r["text"][:300])
    print("------")