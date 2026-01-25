from pinecone import Pinecone
import cohere, os
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

emb = co.embed(
    texts=["This is a test document"],
    model="embed-english-v3.0",
    input_type="search_document",
    embedding_types=["float"]
)

embeddings = emb.embeddings.float
print(len(embeddings[0]))

assert isinstance(embeddings[0], list)
assert isinstance(embeddings[0][0], float)

index.upsert(
    vectors=[
        {
        "id": "doc1",
        "values": embeddings[0],
        "metadata": {"text": "This is a test document"}
        }
    ]
)

query_emb = co.embed(
    texts=["test document"],
    model="embed-english-v3.0",
    input_type="search_query",
    embedding_types=["float"]
)

query_vector = query_emb.embeddings.float[0]

results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True
)

print(results)
