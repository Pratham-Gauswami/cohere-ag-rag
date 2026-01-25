# Testing cohere embeddings
import cohere, os
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))

emb = co.embed(
    texts=["Hello world"],
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float"]
)

#print(len(emb.embeddings[0]))
embeddings = emb.embeddings.float

print(f"Embedding length: {len(embeddings[0])}")