# RAG Search Engine

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## Usage

### Option 1: Use the sample dataset

1. **Download sample data**: Download the sample movies dataset:
   ```bash
   wget https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json -O data/movies.json

3. **Search the sample data**:
   ```bash
   uv run cli/keyword_search_cli.py search "your search query"
   ```

## Configuration

The search engine is configured in `cli/lib/search_utils.py`:
- `DATA_PATH`: Path to your JSON dataset
- `STOP_WORDS_PATH`: Path to stopwords file
- `DEFAULT_SEARCH_LIMIT`: Maximum number of results to return
