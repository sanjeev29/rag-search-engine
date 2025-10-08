import string

from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

stemmer = PorterStemmer()


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
    