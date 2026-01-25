import os
import cohere
from pinecone import Pinecone
from dotenv import load_dotenv

# Load API keys from .env
load_dotenv()

# Initialize Cohere client for embeddings and generation
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Initialize Pinecone client and connect to your index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")  # make sure this matches your .env
index = pc.Index(index_name)

# -------------------------------
# Function: Retrieve vectors from Pinecone
# -------------------------------
def retrieve_vectors(query, top_k=5):
    """
    Takes a user query, embeds it using Cohere, and returns the top_k relevant vectors from Pinecone.
    """
    # Embed the query
    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=['float']
    )
    query_embedding = response.embeddings.float[0]

    # Query Pinecone index
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    return results['matches']

# -------------------------------
# Function: Build prompt for generation
# -------------------------------
def build_prompt(query, matches):
    """
    Converts retrieved vectors into a context string for the LLM and combines with the user query.
    """
    context_texts = []
    for m in matches:
        metadata = m['metadata']
        context_texts.append(
            f"County: {metadata.get('county','N/A')}, "
            f"Crop: {metadata.get('crop','N/A')}, "
            f"Yield: {metadata.get('yield','N/A')}, "
            f"Latitude: {metadata.get('lat','N/A')}, "
            f"Longitude: {metadata.get('lon','N/A')}"
        )

    context = "\n".join(context_texts)

    prompt = (
        f"Answer the following question based on the data below:\n\n"
        f"{context}\n\n"
        f"Question: {query}\nAnswer:"
    )
    return prompt

# -------------------------------
# Function: Generate answer from Cohere
# -------------------------------
def generate_answer(prompt):
    """
    Sends the prompt to Cohere's Chat API to generate a response.
    """
    response = co.chat(
        model="command-a-03-2025",  # or "command-xlarge-nightly" if available for chat
        message=prompt,
        max_tokens=200,
        temperature=0
    )
    #return response.message.content.strip()
    return response.text.strip()

# -------------------------------
# Main RAG flow
# -------------------------------
if __name__ == "__main__":
    # Get user query
    query = input("Ask a question about corn data: ")

    # Step 1: Retrieve relevant vectors from Pinecone
    matches = retrieve_vectors(query)

    if not matches:
        print("No relevant data found for your query.")
    else:
        # Step 2: Build prompt from retrieved context
        prompt = build_prompt(query, matches)

        # Step 3: Generate answer from Cohere
        answer = generate_answer(prompt)

        # Step 4: Display the answer
        print("\nAnswer:\n", answer)
