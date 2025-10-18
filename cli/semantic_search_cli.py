#!/usr/bin/env python3

import argparse

from lib.search_utils import DEFAULT_SEARCH_LIMIT
from lib.semantic_search import (
    embed_query_text_command,
    embed_text_command,
    search_command,
    verify_embeddings_command,
    verify_model_command
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
    embed_query_parser = subparsers.add_parser("embedquery", help="Generate query embedding")
    embed_query_parser.add_argument("query", type=str, help="Query text")

    # Semantic Search
    search_parser = subparsers.add_parser("search", help="Search movies by Semantic Search scoring")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit (Optional)")

    args = parser.parse_args()

    match args.command:
        case "embedquery":
            embed_query_text_command(args.query)
        case "embed_text":
            embed_text_command(args.text)
        case "search":
            results = search_command(args.query, args.limit)

            for i, result in enumerate(results, start=1):
                print(f"{i}.\t{result['title']} (score: {result['score']:.2f})")
                print(f"\t{result['description'][:50]}...")

        case "verify":
            verify_model_command()
        case "verify_embeddings":
            verify_embeddings_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()