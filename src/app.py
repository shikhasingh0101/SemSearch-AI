"""Streamlit user interface for SemSearch AI."""
import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from semantic_search import SemSearch

st.set_page_config(page_title="SemSearch AI", page_icon="🔎", layout="wide")
st.title("🔎 SemSearch AI")
st.caption("Find internships, courses, jobs and research topics by meaning—not only exact keywords.")

@st.cache_resource(show_spinner="Loading the embedding model and building the FAISS index...")
def load_engine():
    return SemSearch()

try:
    engine = load_engine()
except Exception as exc:
    st.error("The search engine could not start. Check that dependencies are installed and the embedding model can be downloaded.")
    st.exception(exc)
    st.stop()

query = st.text_input("What are you looking for?", "machine learning internship")
top_k = st.slider("Number of results", 1, 10, 5)
mode = st.radio("Search mode", ["Semantic search", "Keyword search", "Compare both"], horizontal=True)

if st.button("Search", type="primary") and query.strip():
    if mode in ("Semantic search", "Compare both"):
        st.subheader("Semantic results — Sentence Transformers + FAISS")
        results = engine.semantic_search(query, top_k)
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {row['title']}**  ·  `{row['category']}`")
                st.caption(f"Record: {row['record_id']}  |  Cosine similarity: {float(row['similarity']):.4f}")
                st.write(row["description"])
    if mode in ("Keyword search", "Compare both"):
        st.subheader("Keyword results — TF-IDF baseline")
        results = engine.keyword_search(query, top_k)
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {row['title']}**  ·  `{row['category']}`")
                st.caption(f"Record: {row['record_id']}  |  TF-IDF cosine: {float(row['similarity']):.4f}")
                st.write(row["description"])

st.divider()
st.markdown("**Try meaning-based queries**")
st.write("- machine learning internship")
st.write("- AI trainee opportunities")
st.write("- learn how computers understand human language")
st.write("- find a beginner role building intelligent software")
st.caption("The semantic model is downloaded on first run. Similarity scores are relevance scores, not probabilities.")
