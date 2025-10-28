#!/usr/bin/env python3

import argparse

from lib.search_utils import (
    DEFAULT_MAX_CHUNK_SIZE,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE
)
from lib.semantic_search import (
    chunk_text_command,
    embed_query_text_command,
    embed_text_command,
    search_command,
    semantic_chunk_command,
    verify_embeddings_command,
    verify_model_command
)
from lib.chunked_semantic_search import (
    embed_chunks_command,
    search_chunked_command
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Verify model
    subparsers.add_parser("verify", help="Verify the model used for Semantic Search")
    
    # Generate embedding
    embed_parser = subparsers.add_parser("embed_text", help="Generate an embedding for a single text input")
    embed_parser.add_argument("text", type=str, help="Single text")

    # Verify embeddings
    subparsers.add_parser("verify_embeddings", help="Verify movie dataset embeddings")

    # Generate query embedding
    embed_query_parser = subparsers.add_parser("embed_query", help="Generate query embedding")
    embed_query_parser.add_argument("query", type=str, help="Query text")

    # Semantic Search
    search_parser = subparsers.add_parser("search", help="Search movies by Semantic Search scoring")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit (Optional)")

    # Text chunk
    chunk_parser = subparsers.add_parser("chunk", help="Splits long text into smaller text of given chunk size")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Fixed chunk size (Optional)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Chunk overlap size (Optional)")

    # Semantic Text chunk
    chunk_parser = subparsers.add_parser("semantic_chunk", help="Splits text on sentence boundaries to preserve meaning")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_MAX_CHUNK_SIZE, help="Fixed chunk size (Optional)")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Chunk overlap size (Optional)")

    # Embed chunks
    embed_chunk = subparsers.add_parser("embed_chunks", help="Build chunk embeddings for the movies dataset")

    # Chunked Semantic Search
    search_chunked = subparsers.add_parser("search_chunked", help="")
    search_chunked.add_argument("query", type=str, help="Search query")
    search_chunked.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit (Optional)")

    args = parser.parse_args()

    match args.command:
        case "chunk":
            chunks = chunk_text_command(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")

            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "embed_chunks":
            embed_chunks_command()
        case "embed_query":
            embed_query_text_command(args.query)
        case "embed_text":
            embed_text_command(args.text)
        case "search":
            results = search_command(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"{i}.\t{result['title']} (score: {result['score']:.2f})")
                print(f"\t{result['description'][:200]}...")
        case "search_chunked":
            results = search_chunked_command(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['description']}...")
        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")
        case "verify":
            verify_model_command()
        case "verify_embeddings":
            verify_embeddings_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()