import os
from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever
from src.reranker import Reranker

def main():
    print("Loading retrievers and chunks...")
    chunk_path = "data/chunks.json"
    
    dense = DenseRetriever()
    dense.load_chunks(chunk_path)
    
    sparse = SparseRetriever()
    sparse.load_chunks(chunk_path)
    
    hybrid = HybridRetriever(dense, sparse, alpha=0.6)
    
    query = "What are the common side effects of ACE inhibitors?"
    print(f"\nQuery: {query}")
    
    print("Performing hybrid search (k=20)...")
    results = hybrid.search(query, k=20)
    print(f"Found {len(results)} initial results.")
    
    print("\nInitializing Reranker...")
    try:
        reranker = Reranker()
        print("Reranking results (top_k=5)...")
        final_results = reranker.rerank(query, results, top_k=5)
        
        print("\n--- FINAL RERANKED RESULTS ---")
        for i, res in enumerate(final_results):
            score = res.get('rerank_score', 'N/A')
            print(f"Rank {i+1} | Rerank Score: {score:.4f} | Chunk ID: {res.get('chunk_id')}")
            print(f"Snippet: {res['text'][:150]}...\n")
            
        print("Reranker is working successfully!")
        
    except Exception as e:
        print(f"Error during reranking: {e}")

if __name__ == "__main__":
    main()
