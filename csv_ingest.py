import csv
import os
import time
import cohere
from dotenv import load_dotenv
from pinecone import Pinecone

# Load API keys and environment variables from .env
load_dotenv()

# Initialize clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Function to convert a row into a meaningful text block for embedding
def row_to_text(row):
    return (
        f"A farmer in {row['County']} grows {row['Crop']} on {row['Acreage']} acres "
        f"with a yield of {row['Yield']}. "
        f"They use {row['Water source']} as their water source and "
        f"{row['Power source']} as their power source. "
        f"The fertilizer amount used is {row['Fertilizer amount']} and "
        f"{row['Laborers']} laborers are employed. "
        f"Agricultural advice comes from {row['Main advisory source']} "
        f"provided by {row['Extension provider']} "
        f"via {row['Advisory format']} in {row['Advisory language']}."
    )

# Read CSV into a list of tuples: (row_index, row_dict)
rows = []
with open("corn_data.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        rows.append((i, row))

# Batch embedding to avoid exceeding Cohere trial rate limit
batch_size = 20   # number of rows per API call
delay_seconds = 2  # wait time between batches to prevent 429 errors
vectors = []

# Process in batches
for i in range(0, len(rows), batch_size):
    batch_rows = rows[i:i+batch_size]
    texts = [row_to_text(r[1]) for r in batch_rows]

    # Get embeddings for the batch
    emb_batch = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"]
    )

    # Add each embedding to vectors list
    for j, r in enumerate(batch_rows):
        vectors.append({
            "id": f"row-{r[0]}",
            "values": emb_batch.embeddings.float[j],
            "metadata": {
                "county": r[1]["County"],
                "crop": r[1]["Crop"],
                "yield": r[1]["Yield"],
                "lat": r[1]["Latitude"],
                "lon": r[1]["Longitude"]
            }
        })

    print(f"Processed batch {i // batch_size + 1} / {((len(rows)-1)//batch_size)+1}")
    time.sleep(delay_seconds)  # prevent 429 Too Many Requests

# Upsert all vectors at once
index.upsert(vectors=vectors)
print("CSV data successfully ingested into Pinecone")
