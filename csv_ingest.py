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
    # Helper function to handle empty values
    def get_value(key, default='unknown'):
        value = row.get(key, default)
        return value if value and value.strip() else default
    
    return (
        f"Farmer {get_value('Farmer')} in {get_value('County')} county grows {get_value('Crop')} "
        f"on {get_value('Acreage')} acres with a total yield of {get_value('Yield')} bushels. "
        f"Education: {get_value('Education')}, Gender: {get_value('Gender')}, "
        f"Age: {get_value('Age bracket')}, Household size: {get_value('Household size')}. "
        f"Farm uses {get_value('Fertilizer amount')} units of fertilizer and employs {get_value('Laborers')} laborers. "
        f"Water source: {get_value('Water source')}, Power source: {get_value('Power source')}. "
        f"Credit source: {get_value('Main credit source')}, Crop insurance: {get_value('Crop insurance')}, "
        f"Farm records maintained: {get_value('Farm records')}. "
        f"Agricultural advice from {get_value('Main advisory source')} "
        f"provided by {get_value('Extension provider')} "
        f"via {get_value('Advisory format')} in {get_value('Advisory language')}."
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
                "farmer": r[1].get("Farmer", "Unknown"),
                "county": r[1].get("County", "Unknown"),
                "crop": r[1].get("Crop", "Unknown"),
                "yield": r[1].get("Yield", "0"),
                "acreage": r[1].get("Acreage", "0"),
                "education": r[1].get("Education", "Unknown"),
                "gender": r[1].get("Gender", "Unknown"),
                "age_bracket": r[1].get("Age bracket", "Unknown"),
                "household_size": r[1].get("Household size", "0"),
                "fertilizer_amount": r[1].get("Fertilizer amount", "0"),
                "laborers": r[1].get("Laborers", "0"),
                "water_source": r[1].get("Water source", "Unknown"),
                "power_source": r[1].get("Power source", "Unknown"),
                "credit_source": r[1].get("Main credit source", "Unknown"),
                "crop_insurance": r[1].get("Crop insurance", "Unknown"),
                "farm_records": r[1].get("Farm records", "Unknown"),
                "advisory_source": r[1].get("Main advisory source", "Unknown"),
                "extension_provider": r[1].get("Extension provider", "Unknown"),
                "advisory_format": r[1].get("Advisory format", "Unknown"),
                "advisory_language": r[1].get("Advisory language", "Unknown"),
                "latitude": r[1].get("Latitude", "0"),
                "longitude": r[1].get("Longitude", "0")
            }
        })

    print(f"Processed batch {i // batch_size + 1} / {((len(rows)-1)//batch_size)+1}")
    time.sleep(delay_seconds)  # prevent 429 Too Many Requests

# Upsert all vectors at once
index.upsert(vectors=vectors)
print("CSV data successfully ingested into Pinecone")
