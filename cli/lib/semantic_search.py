from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')


def verify_model_command():
    search = SemanticSearch()
    print(f"MODEL: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
