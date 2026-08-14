"""
retrieval.py

Sets up a FAISS vector index over the preprocessed interview question bank
and provides a function to retrieve the most semantically similar questions
(with their ideal answers) for a given user question — the "R" in RAG.

Place this file at: src/data_processing/retrieval.py
"""

import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# --- Paths (adjust if your project root differs) ---
TRAIN_CSV_PATH = "data/processed/text/train.csv"
EMBEDDINGS_PATH = "data/processed/text/train_question_embeddings.npy"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


class QuestionRetriever:
    """
    Wraps a FAISS index over question embeddings for semantic retrieval,
    with optional domain filtering (e.g. 'HR' vs 'Technical').
    """

    def __init__(self, train_csv_path=TRAIN_CSV_PATH, embeddings_path=EMBEDDINGS_PATH):
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(
                f"Could not find {train_csv_path}. "
                f"Make sure you're running from the project root and data/ is set up."
            )

        self.train_df = pd.read_csv(train_csv_path)
        self.question_embeddings = np.load(embeddings_path)
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME)

        self.embedding_dim = self.question_embeddings.shape[1]

        # Full index (all domains)
        self.full_index = faiss.IndexFlatL2(self.embedding_dim)
        self.full_index.add(self.question_embeddings.astype("float32"))

        # Pre-build per-domain indexes for fast repeated lookups
        self._domain_indexes = {}
        self._domain_dfs = {}
        for domain in self.train_df["domain"].unique():
            mask = (self.train_df["domain"] == domain).values
            filtered_embeddings = self.question_embeddings[mask]
            domain_index = faiss.IndexFlatL2(self.embedding_dim)
            domain_index.add(filtered_embeddings.astype("float32"))
            self._domain_indexes[domain] = domain_index
            self._domain_dfs[domain] = self.train_df[mask].reset_index(drop=True)

    def retrieve(self, user_question: str, top_k: int = 3, domain: str = None) -> pd.DataFrame:
        """
        Retrieve the top_k most similar questions (with ideal answers) to user_question.

        Args:
            user_question: the question being asked / evaluated against
            top_k: number of similar questions to retrieve
            domain: optional filter, e.g. "HR" or "Technical". If None, searches all.

        Returns:
            DataFrame with columns [question, answer, category, role, difficulty, distance]
        """
        query_embedding = self.embed_model.encode([user_question]).astype("float32")

        if domain and domain in self._domain_indexes:
            distances, indices = self._domain_indexes[domain].search(query_embedding, top_k)
            source_df = self._domain_dfs[domain]
        else:
            distances, indices = self.full_index.search(query_embedding, top_k)
            source_df = self.train_df

        results = source_df.iloc[indices[0]][
            ["question", "answer", "category", "role", "difficulty"]
        ].copy()
        results["distance"] = distances[0]
        return results.reset_index(drop=True)


# Convenience singleton so other modules can just do:
#   from src.data_processing.retrieval import get_retriever
#   get_retriever().retrieve("...", domain="HR")
_retriever_instance = None

def get_retriever():
    """Lazily initializes and returns a shared QuestionRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = QuestionRetriever()
    return _retriever_instance