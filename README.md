# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine implementing multiple search paradigms for semantic and keyword-based information retrieval.

## Features

### Search Algorithms
- **BM25 Keyword Search**: Probabilistic ranking function with tunable parameters
- **Semantic Search**: Vector-based similarity search using sentence transformers
- **Chunked Semantic Search**: Document chunking with semantic understanding for better precision
- **TF-IDF Scoring**: Traditional term frequency-inverse document frequency calculations

## Quick Start

### Installation

```
# Clone the repository
git clone <your-repo-url>
cd rag-search-engine

# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Download Sample Data

```
# Download the movie dataset
wget https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json -O data/movies.json
```

### Build Search Indices

```
# Build keyword search index
uv run cli/keyword_search_cli.py build

# Generate semantic embeddings
uv run cli/semantic_search_cli.py verify_embeddings

# Generate chunked embeddings
uv run cli/semantic_search_cli.py embed_chunks
```

## Usage Examples

### Keyword Search (BM25)

```
# Basic keyword search
uv run cli/keyword_search_cli.py search "action thriller"

# Advanced BM25 search with scoring
uv run cli/keyword_search_cli.py bm25search "space adventure" --limit 10

# Analyze term frequencies
uv run cli/keyword_search_cli.py tf 123 "adventure"
uv run cli/keyword_search_cli.py idf "thriller"
```

### Semantic Search

```
# Semantic similarity search
uv run cli/semantic_search_cli.py search "movies about artificial intelligence"

# Generate embeddings for custom text
uv run cli/semantic_search_cli.py embed_text "A sci-fi movie about robots"

# Chunked semantic search for better precision
uv run cli/semantic_search_cli.py search_chunked "psychological thriller with plot twists"
```

### Text Processing

```
# Fixed-size text chunking
uv run cli/semantic_search_cli.py chunk "Your long text here..." --chunk-size 200 --overlap 50

# Semantic boundary-aware chunking
uv run cli/semantic_search_cli.py semantic_chunk "Your text..." --max-chunk-size 4 --overlap 1

## 🔧 Configuration

Key parameters can be adjusted in `cli/lib/search_utils.py`:

- **BM25 Parameters**: `BM25_K1` (saturation), `BM25_B` (length normalization)
- **Chunking**: `DEFAULT_CHUNK_SIZE`, `DEFAULT_MAX_CHUNK_SIZE`
- **Search Limits**: `DEFAULT_SEARCH_LIMIT`
- **Paths**: Dataset, stopwords, and cache directory locations

## Future Search Enhancements:

[ ] **Hybrid Search**: Combines BM25 keyword scoring with semantic similarity for optimal relevance ranking
[ ] **LLM-Powered Query Rewriting and Expansion**: Uses large language models to expand and refine user queries 
      before search execution
[ ] **Two-Stage Reranking**: Initial broad retrieval followed by precise reranking using advanced scoring 
      algorithms
