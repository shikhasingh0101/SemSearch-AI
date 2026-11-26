
"""
SemSearch AI
Semantic Search: Sentence Transformers + FAISS
Keyword Search: TF-IDF + Cosine Similarity
"""

# Configure native-library threads BEFORE importing ML libraries.
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path

import streamlit as st
import torch
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

torch.set_num_threads(1)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "records.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# --------------------------------------------------
# SEMANTIC SEARCH ENGINE
# --------------------------------------------------

class SemSearch:

    def __init__(self, data_path=DATA_PATH, model_name=MODEL_NAME):

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {data_path}"
            )

        self.df = pd.read_csv(data_path).fillna("")

        required_columns = [
            "record_id",
            "title",
            "category",
            "description",
            "text"
        ]

        missing = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing dataset columns: {missing}"
            )

        self.texts = self.df["text"].astype(str).tolist()

        # Load embedding model on CPU.
        self.model = SentenceTransformer(
    model_name,
    device="cpu",
    model_kwargs={
        "low_cpu_mem_usage": False
    }
)

        # Generate embeddings in small batches.
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        # Build FAISS cosine similarity index.
        self.index = faiss.IndexFlatIP(
            self.embeddings.shape[1]
        )

        self.index.add(self.embeddings)

        # Build TF-IDF keyword baseline.
        self.tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.tfidf.fit_transform(
            self.texts
        )

    # --------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------

    def semantic_search(self, query, top_k=5):

        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")

        scores, ids = self.index.search(
            query_embedding,
            min(top_k, len(self.df))
        )

        results = self.df.iloc[ids[0]].copy()

        results.insert(
            0,
            "similarity",
            scores[0]
        )

        return results

    # --------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------

    def keyword_search(self, query, top_k=5):

        query_vector = self.tfidf.transform([query])

        scores = cosine_similarity(
            query_vector,
            self.tfidf_matrix
        ).ravel()

        ids = np.argsort(scores)[::-1][:top_k]

        results = self.df.iloc[ids].copy()

        results.insert(
            0,
            "similarity",
            scores[ids]
        )

        return results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

def display_results(results, heading):

    st.subheader(heading)

    if results.empty:
        st.info("No results found.")
        return

    for rank, (_, row) in enumerate(
        results.iterrows(), start=1
    ):

        with st.expander(
            f"{rank}. {row['title']} — {row['category']}",
            expanded=(rank == 1)
        ):

            st.write(f"**Record ID:** {row['record_id']}")

            st.write(
                f"**Similarity Score:** "
                f"{float(row['similarity']):.4f}"
            )

            st.write(
                f"**Description:** {row['description']}"
            )


# --------------------------------------------------
# MAIN STREAMLIT APPLICATION
# --------------------------------------------------

def main():

    st.set_page_config(
        page_title="SemSearch AI",
        page_icon="🔎",
        layout="wide"
    )

    st.title("🔎 SemSearch AI")

    st.markdown(
        """
        ### Semantic Search and Recommendation System

        Search using natural language and compare:

        - **Semantic Search:** Sentence Transformers + FAISS
        - **Keyword Search:** TF-IDF + Cosine Similarity
        """
    )

    st.divider()

    # Load the search engine once for this diagnostic run.
    try:
        with st.spinner(
            "Loading dataset and AI embedding model..."
        ):
            engine = SemSearch()

        st.success(
            f"Search engine ready! {len(engine.df)} records loaded."
        )

    except Exception as error:
        st.error(
            f"Could not initialize the search engine: {error}"
        )
        st.exception(error)
        st.stop()

    # Sidebar controls.
    st.sidebar.header("Search Settings")

    search_mode = st.sidebar.selectbox(
        "Choose Search Method",
        [
            "Semantic Search",
            "Keyword Search",
            "Compare Both"
        ]
    )

    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=min(10, len(engine.df)),
        value=min(5, len(engine.df))
    )

    query = st.text_input(
        "Enter your search query",
        placeholder="Example: AI trainee opportunities"
    )

    search_button = st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True
    )

    if search_button:

        if not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching records..."):

            if search_mode == "Semantic Search":

                results = engine.semantic_search(
                    query, top_k
                )

                display_results(
                    results,
                    "🧠 Semantic Search Results"
                )

            elif search_mode == "Keyword Search":

                results = engine.keyword_search(
                    query, top_k
                )

                display_results(
                    results,
                    "📝 Keyword Search Results"
                )

            else:

                semantic_results = engine.semantic_search(
                    query, top_k
                )

                keyword_results = engine.keyword_search(
                    query, top_k
                )

                col1, col2 = st.columns(2)

                with col1:
                    display_results(
                        semantic_results,
                        "🧠 Semantic Search"
                    )

                with col2:
                    display_results(
                        keyword_results,
                        "📝 Keyword Search"
                    )

    # Dataset preview.
    st.divider()

    with st.expander("📚 View Dataset"):

        st.write(f"Total records: {len(engine.df)}")

        st.table(engine.df)
    st.caption(
        "SemSearch AI | Group 3 | "
        "Sentence Transformers + FAISS + TF-IDF"
    )


if __name__ == "__main__":
    main()
