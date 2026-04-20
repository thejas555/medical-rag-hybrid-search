import streamlit as st
import os
import sys
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
print("DEBUG API KEY:", os.getenv("GROQ_API_KEY"))

from src.dense_retriever import DenseRetriever
from src.sparse_retriever import SparseRetriever
from src.hybrid_retriever import HybridRetriever
from src.generator import Generator


st.set_page_config(page_title="Medical RAG Assistant", layout="wide")

st.title("Medical Research RAG Assistant")

st.write(
    "Ask questions about medical research abstracts using a Hybrid Retrieval-Augmented Generation system."
)
#Slider to adjust weight
alpha = st.slider(
    "Dense vs BM25 Weight (α)",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
)

st.write(f"Hybrid score formula: α * dense + (1-α) * bm25")

@st.cache_resource(show_spinner="Loading Retrieval System...")
def load_system():

    chunk_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'data', 'chunks.json')
    )

    dense = DenseRetriever()
    dense.load_chunks(chunk_path)

    sparse = SparseRetriever()
    sparse.load_chunks(chunk_path)

    generator = Generator()

    return dense, sparse, generator

from dotenv import load_dotenv
load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    st.error("GROQ_API_KEY environment variable not set. Please set it in your .env file!")
    st.stop()

dense, sparse, generator = load_system()

hybrid = HybridRetriever(dense, sparse, alpha=alpha)

query = st.text_input("Enter your medical question:")

if query:
    st.write("### Retrieving Sources...")
    #results = hybrid.search(query, k=5)
    
    #replacing above line with script to get reranked results.
    from src.reranker import Reranker
    reranker = Reranker()
    results = hybrid.search(query, k=20)
    final_results = reranker.rerank(query, results, top_k=5)

    st.write("### Retrieval Analysis")

    query_embedding = hybrid.dense.model.encode([query])

    doc_embeddings = []

    for r in results:
        idx = r["chunk_id"]
        emb = hybrid.dense.model.encode([r["text"]])[0]
        doc_embeddings.append(emb)

    doc_embeddings = np.array(doc_embeddings)

    import pandas as pd

    table = []

    for rank, r in enumerate(results, start=1):
        table.append({
            "Rank": rank,
            "Chunk ID": r.get("chunk_id", ""),
            "Dense Score": r.get("dense_score", 0),
            "BM25 Score": r.get("bm25_score", 0),
            "Dense Norm": r.get("dense_norm", 0),
            "BM25 Norm": r.get("bm25_norm", 0),
            "Hybrid Score": r.get("final_score", r.get("score", 0))
        })

    df = pd.DataFrame(table)
    st.dataframe(df)

    # Hybrid Score Visualization
    st.write("### Hybrid Ranking Scores")
    chart_data = df.set_index("Rank")["Hybrid Score"]
    st.bar_chart(chart_data)



    #initially the plot wasn't using normalized values, changed it
    st.write("### Dense vs BM25 Contributions (Normalized)")
    dense_vals = df.set_index("Rank")["Dense Norm"]
    bm25_vals = df.set_index("Rank")["BM25 Norm"]

    chart_df = pd.DataFrame({
        "Dense": dense_vals,
        "BM25": bm25_vals
    })
    st.bar_chart(chart_df)



    # Highlight Query Tokens in Sources
    query_terms = query.lower().split()

    for rank, r in enumerate(results, start=1):
        with st.expander(f"Source {rank} | Chunk {r.get('chunk_id', '')}"):
            text = r["text"]

            for term in query_terms:
                text = text.replace(term, f"**{term}**")

            st.write(text)
            st.write("**Score Breakdown:**")
            st.write(f"Dense score: {r.get('dense_score', 'N/A')}")
            st.write(f"BM25 score: {r.get('bm25_score', 'N/A')}")
            st.write(
                f"Hybrid score = {alpha:.2f} * {r['dense_norm']:.3f} + {(1-alpha):.2f} * {r['bm25_norm']:.3f} = {r['final_score']:.3f}"
                )

    st.write("### Generating Answer...")

    #contexts = [{"text": r["text"]} for r in results]
    #change made for reranker
    contexts = [{"text": r["text"]} for r in final_results]

    answer = generator.generate(query, contexts)

    st.write("### Answer")
    st.write(answer)

    #Run PCA
    # Combine query + docs
    all_embeddings = np.vstack([query_embedding, doc_embeddings])

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_embeddings)

    query_point = reduced[0]
    doc_points = reduced[1:]

    # By default, matplotlib uses (6.4, 4.8). We scale it down to roughly 1/3 the area.
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # plot documents
    ax.scatter(doc_points[:,0], doc_points[:,1], label="Retrieved Docs")

    # plot query
    ax.scatter(query_point[0], query_point[1], marker="*", s=200, label="Query")

    for i, point in enumerate(doc_points):
        ax.text(point[0], point[1], f"D{i+1}")

    ax.legend()
    ax.set_title("Embedding Space (PCA Projection)")

    # use_container_width=False prevents Streamlit from blowing the small plot back up to full width
    st.pyplot(fig, use_container_width=False)