"""
SemSearch AI: semantic search using sentence embeddings + FAISS cosine similarity.
Run from the project root:
    python src/semantic_search.py --query "machine learning internship" --top-k 5
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "records.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class SemSearch:
    def __init__(self, data_path=DATA_PATH, model_name=MODEL_NAME):
        self.df = pd.read_csv(data_path).fillna("")
        self.model = SentenceTransformer(model_name)
        self.texts = self.df["text"].astype(str).tolist()
        # Normalize vectors so inner product is equivalent to cosine similarity.
        self.embeddings = self.model.encode(
            self.texts, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        # Traditional keyword baseline using TF-IDF and cosine similarity.
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.texts)

    def semantic_search(self, query, top_k=5):
        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(top_k, len(self.df)))
        results = self.df.iloc[ids[0]].copy()
        results.insert(0, "similarity", scores[0])
        return results

    def keyword_search(self, query, top_k=5):
        q = self.tfidf.transform([query])
        scores = cosine_similarity(q, self.tfidf_matrix).ravel()
        ids = np.argsort(scores)[::-1][:top_k]
        results = self.df.iloc[ids].copy()
        results.insert(0, "similarity", scores[ids])
        return results

def show(label, results):
    print("\n" + "=" * 88)
    print(label)
    print("=" * 88)
    for rank, (_, row) in enumerate(results.iterrows(), 1):
        print(f"{rank}. [{row['record_id']}] {row['title']} | {row['category']}")
        print(f"   Score: {float(row['similarity']):.4f}")
        print(f"   {row['description'][:210]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="machine learning internship")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    engine = SemSearch()
    show("SEMANTIC SEARCH (Sentence Transformer + FAISS)", engine.semantic_search(args.query, args.top_k))
    show("KEYWORD BASELINE (TF-IDF + cosine similarity)", engine.keyword_search(args.query, args.top_k))

if __name__ == "__main__":
    main()
