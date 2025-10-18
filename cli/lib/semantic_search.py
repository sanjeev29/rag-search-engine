import os
from unittest import result
import numpy as np

from typing import Any
from sentence_transformers import SentenceTransformer

from lib.search_utils import (
    CACHE_DIR, 
    load_movies
)

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def build_embeddings(self, documents: list[dict]) -> list[Any]:
        self.documents = documents

        movies_to_embed = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            movies_to_embed.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movies_to_embed, show_progress_bar=True)

        # Save embeddings to a file
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def generate_embedding(self, text: str):
        text = text.strip()

        if text == "":
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode([text])[0]

        return embedding

    def load_or_create_embeddings(self, documents: list[dict]) -> list[Any]:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        # If embeddings are cached, load and return them
        if os.path.exists(self.embeddings_path):
            self.embeddings = np.load(self.embeddings_path, 'r')
            if len(self.embeddings) == len(documents):
                return self.embeddings

        # Build embeddings for the given documents and cache them
        return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict]:
        # Check if embeddings are loaded
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        # Generate embedding for given query
        query_embedding = self.generate_embedding(text=query)

        # Calculate cosine similarity between query embedding and each document embedding
        scores = []
        for i, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, embedding)
            scores.append((similarity_score, self.documents[i]))

        # Sort by similarity score
        scores.sort(key=lambda x:x[0], reverse=True)

        return [
            {"title": doc['title'], "description": doc['description'], "score": score} 
            for score, doc in scores[:limit]
        ]


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def embed_query_text_command(query: str):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def embed_text_command(text: str):
    search = SemanticSearch()

    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def search_command(query: str, limit: int) -> list[dict]:
    search = SemanticSearch()
    docs = load_movies()
    search.load_or_create_embeddings(docs)

    return search.search(query, limit)


def verify_embeddings_command():
    search = SemanticSearch()
    documents = load_movies()
    embeddings = search.load_or_create_embeddings(documents)

    print(f"Number of docs: {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")


def verify_model_command():
    search = SemanticSearch()
    print(f"MODEL: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
