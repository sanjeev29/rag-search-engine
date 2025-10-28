import json
import os
from typing import Any
import numpy as np

from .semantic_search import (
    semantic_chunk_command,
    SemanticSearch
)
from .search_utils import (
    CACHE_DIR, 
    DEFAULT_MAX_CHUNK_SIZE, 
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
        self.chunk_metadata = document_chunk_metadata

        # Save chunk embeddings to a file
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)

        # Save chunk metadata to a json file
        with open(self.chunk_metadata_path, 'w') as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(document_chunks)}, f, indent=2)

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

        if not self.chunk_embeddings or not self.chunk_metadata:
            self.build_chunk_embeddings(documents)

        return self.chunk_embeddings


def embed_chunks_command():
    documents = load_movies()
    chunkedSS = ChunkedSemanticSearch()
    
    # Load or build chunk embeddings for the given documents
    embeddings = chunkedSS.load_or_create_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")
