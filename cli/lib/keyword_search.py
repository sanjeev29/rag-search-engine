import os
import pickle
import string
from collections import defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

stemmer = PorterStemmer()

class InvertedIndex:
    def __init__(self) -> None:
        # Maps tokens -> document IDs
        self._index = defaultdict(set)

        # Maps document IDs -> full document objects
        self._docmap: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        # Tokenize input text
        tokens = tokenize_text(text)

        # Add each token to the index with document ID
        for token in tokens:
            self._index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self._index.get(term, set())

        return sorted(list(doc_ids))

    def build(self) -> None:
        # Read movies data
        movies = load_movies()

        # Iterate over all movies and add them to both index and docmap
        for movie in movies:
            # Store the full document in docmap
            self._docmap[movie['id']] = movie
            
            # Add to inverted index
            input_text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id=movie['id'], text=input_text)

    def save(self) -> None:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Save inverted index
        with open(self.index_path, 'wb') as f:
            pickle.dump(self._index, f)
        
        # Save document map
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self._docmap, f)


def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    results = []
    
    # Preprocess query
    query_tokens = tokenize_text(query)
    
    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        
        # Check if any query word matches any title word
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break
    
    return results


def preprocess_text(text: str) -> list[str]:
    """Preprocess text for text matching."""
    # Convert to lowercase
    text = text.lower()

    # Create a translation table that maps all punctuation to None (removes them)
    translator = str.maketrans('', '', string.punctuation)
    # Remove all punctuation from text
    text = text.translate(translator)
    
    return text


def tokenize_text(text: str) -> list[str]:
    """Split text and convert it into word-based tokens, filtering out stopwords."""
    text = preprocess_text(text)
    
    # Split text into word-based tokens
    tokens = text.split()
    
    # Load stopwords
    stop_words = load_stop_words()
    
    # Filter out empty tokens and stopwords
    return [stemmer.stem(token) for token in tokens if token.strip() and token not in stop_words]


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    
    return False
    