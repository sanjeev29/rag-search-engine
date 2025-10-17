from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def generate_embedding(self, text: str):
        text = text.strip()

        if text == "":
            raise ValueError("Text cannot be empty.")

        embedding = self.model.encode([text])[0]

        return embedding
        

def embed_text_command(text: str):
    search = SemanticSearch()

    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model_command():
    search = SemanticSearch()
    print(f"MODEL: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")
