import json
import os

# BM25 search parameters

# Saturation parameter for the term frequency
BM25_K1 = 1.5

# Normalization strength - controls the impact of document length on relevance scoring
# b = 0 -> no effect
# b = 1 -> full length normalization
BM25_B = 0.75


DEFAULT_SEARCH_LIMIT = 5
DEFAULT_CHUNK_SIZE = 200

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOP_WORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data.get("movies", [])

def load_stop_words() -> list[str]:
    """Load stop words from file and return as a list of strings."""
    with open(STOP_WORDS_PATH, "r", encoding="utf-8") as f:
        # Read all lines and strip whitespace/newlines
        return f.read().splitlines()
