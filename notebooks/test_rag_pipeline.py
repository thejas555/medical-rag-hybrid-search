import os
import sys
from groq import Groq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever


class Generator:

    def __init__(self):

        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )


    def build_prompt(self, query, chunks):

        context = ""

        for i, chunk in enumerate(chunks):
            context += f"[Source {i+1}] {chunk['text']}\n\n"

        prompt = f"""
You are a medical research assistant.

Use ONLY the provided sources to answer the question.

If the answer is not present in the sources, say:
"The provided documents do not contain sufficient information."

Question:
{query}

Sources:
{context}

Answer:
"""

        return prompt


    def generate(self, query, chunks):

        prompt = self.build_prompt(query, chunks)

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return response.choices[0].message.content


if __name__ == "__main__":
    dense = DenseRetriever()
    chunk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks.json'))
    dense.load_chunks(chunk_path)

    sparse = SparseRetriever()
    sparse.load_chunks(chunk_path)

    hybrid = HybridRetriever(dense, sparse)
    generator = Generator()

    query = "What are the common treatments for knee osteoarthritis?"

    print(f"\nQuery: {query}\n")

    print("Retrieving chunks...")

chunks = hybrid.search(query, k=5)

print("\nRetrieved Sources:\n")

for i, chunk in enumerate(chunks):
    print(f"\nSource {i+1}:")
    print(chunk["text"][:300])
    print("------")

print("\nGenerating response...\n")

answer = generator.generate(query, chunks)

print("\n--- Answer ---\n")
print(answer)
print("\n--------------\n")