from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512):
        """
        Initializes the cross-encoder model for reranking.
        """
        self.model = CrossEncoder(model_name, max_length=max_length)

    def rerank(self, query, top_k_results, top_k=5):
        """
        Reranks a list of initial results based on a query.
        
        Args:
            query (str): The search query.
            top_k_results (list): A list of dictionaries, where each dict has a "text" key.
            top_k (int): The number of top results to return after reranking.
            
        Returns:
            list: The top_n reranked results, with an added 'rerank_score' key.
        """
        if not top_k_results:
            return []

        # Prepare pairs for the CrossEncoder: (Query, Document)
        pairs = [[query, result["text"]] for result in top_k_results]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores to results and sort
        for i, score in enumerate(scores):
            top_k_results[i]["rerank_score"] = float(score)
            
        ranked_results = sorted(top_k_results, key=lambda x: x["rerank_score"], reverse=True)
        
        return ranked_results[:top_k]
