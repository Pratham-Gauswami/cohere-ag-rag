import os
import cohere
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

#Take a user query, embed it, retrieve top matches from Pinecone.
def retrieval_vectors(query, top_k=5):
    #query = input("Enter your query to seach the db:")
    # Step 1: Embed the query
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=['float']
    )
    
    query_embedding = response.embeddings.float[0]
    print("query_embedding first 10 values: ", query_embedding[:10] + ["..."])

    # Query Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    # Step 3: Return the results
    return results['matches']
