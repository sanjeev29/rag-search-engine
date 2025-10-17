#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    embed_text_command,
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

    args = parser.parse_args()

    match args.command:
        case "embed_text":
            embed_text_command(args.text)
        case "verify":
            verify_model_command()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()