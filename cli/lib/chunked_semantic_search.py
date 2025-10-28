import json
import os
from typing import Any
from nltk import defaultdict
import numpy as np

from .semantic_search import (
    cosine_similarity,
    semantic_chunk_command,
    SemanticSearch
)
from .search_utils import (
    CACHE_DIR, 
    DEFAULT_MAX_CHUNK_SIZE,
    SCORE_PRECISION, 
    load_movies
)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> list[Any]:
        self.documents = documents
        self.document_map = {document['id']: document for document in documents}
        document_chunks: list[str] = []
        document_chunk_metadata: list[dict] = []

        for document in documents:
            description = document['description']
            if not description.strip():
                continue
            
            # Split description into 4 sentence chunks with 1 sentence overlap
            chunks = semantic_chunk_command(
                text=description, 
                max_chunk_size=DEFAULT_MAX_CHUNK_SIZE,
                overlap=1
            )

            for idx, chunk in enumerate(chunks):
                document_chunks.append(chunk)
                document_chunk_metadata.append({
                    "movie_idx": document['id'],
                    "chunk_idx": idx,
                    "total_chunks": len(chunks)
                })

        self.chunk_embeddings = self.model.encode(document_chunks, show_progress_bar=True)
        self.chunk_metadata = {"chunks": document_chunk_metadata, "total_chunks": len(document_chunks)}

        # Save chunk embeddings to a file
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        # Save chunk metadata to a json file
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> list[Any]:
        self.documents = documents
        self.document_map = {document['id']: document for document in documents}

        # If chunk embeddings are cached, load them
        if os.path.exists(self.chunk_embeddings_path):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path, 'r')

        # If chunk metadata is cached, load them
        if os.path.exists(self.chunk_metadata_path):
            with open(self.chunk_metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)

        if self.chunk_embeddings is None or self.chunk_metadata is None:
            self.build_chunk_embeddings(documents)

        return self.chunk_embeddings

    def search_chunks(self, query: str, limit: int = 10) -> list[dict]:
        # Generate an embedding of the query
        query_embedding = self.generate_embedding(query)
        
        # Track scores for each chunk embedding
        chunk_scores: list[dict] = []
        
        # For each chunk embedding
        for chunk_idx, metadata in enumerate(self.chunk_metadata.get("chunks", [])):
            # Calculate the cosine similarity between the chunk embedding and the query embedding
            chunk_embedding = self.chunk_embeddings[chunk_idx]
            score = cosine_similarity(query_embedding, chunk_embedding)
            
            chunk_scores.append({
                "chunk_idx": metadata['chunk_idx'],  # The index of the chunk within the document
                "movie_idx": metadata["movie_idx"],  # The index of the document in self.documents
                "score": score  # The cosine similarity score
            })
        
        # Track scores for each movie
        movie_scores: dict = {}
        
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        # Sort the movie scores by score in descending order
        sorted_movie_scores = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for movie_idx, score in sorted_movie_scores[:limit]:
            doc = self.document_map.get(movie_idx)
            if doc:
                results.append({
                    "id": doc["id"],
                    "title": doc["title"],
                    "description": doc["description"][:100],
                    "score": round(score, SCORE_PRECISION)
                })
        
        return results


def embed_chunks_command():
    documents = load_movies()
    chunkedSS = ChunkedSemanticSearch()
    embeddings = chunkedSS.load_or_create_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int) -> list[dict]:
    documents = load_movies()
    chunkedSS = ChunkedSemanticSearch()
    _ = chunkedSS.load_or_create_embeddings(documents)

    return chunkedSS.search_chunks(query, limit)