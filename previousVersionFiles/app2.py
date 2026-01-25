import streamlit as st
import cohere
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import pandas as pd
import pydeck as pdk

load_dotenv()

#Initialize clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

st.set_page_config(
    page_title="Agricultural Intelligence Dashboard",
    layout="wide"
)

st.title("🌾 Agricultural Intelligence Dashboard")

st.markdown(
    """
    Ask natural language questions about **corn yield data**  
    using an AI-powered Retrieval-Augmented Generation (RAG) system.
    """
)

st.divider()

st.subheader("Ask a question about corn data")

#Input box
query = st.text_input(
    "Type your question here:",
    placeholder="e.g. Which counties appear in the dataset?"
)

#Button to submit
ask_button = st.button("Get Answer")

#Debug: show input for now
if ask_button:
    st.info(f"You typed: {query}")


#Only run if the answer button is clicked
if ask_button and query:

    response = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query",
        embedding_types=["float"]
    )
    query_embedding = response.embeddings.float[0]

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True
    )

    #Collect context text
    context_texts = []
    for match in results['matches']:
        #Example: combine fields into a readable string
        meta = match['metadata']
        context_texts.append(f"County: {meta.get('county', '')}, Crop: {meta.get('crop', '')}, Yield: {meta.get('Yield', '')}")


    #combine context for LLM
    context_block = "\n".join(context_texts)

    #Generate answer
    prompt = f"""
    Use the following data to answer the question. If the answer is not present, say 'I don't know'.

    Data:
    {context_block}

    Question:
    {query}
    """

    chat_reponse = co.chat(
        model="command-a-03-2025",
        message=prompt,
        temperature=0
    )               

    #Display answer
    answer_text = chat_reponse.text.strip()
    st.subheader("🤖 Answer")  
    st.success(answer_text)

    # --- Step 4: Display Retrieved Context ---
if ask_button and query and results['matches']:
    st.subheader("📄 Retrieved Context")
    
    for i, match in enumerate(results['matches'], start=1):
        meta = match['metadata']
        with st.expander(f"Context {i} (Score: {match['score']:.2f})"):
            st.markdown(f"""
            **County:** {meta.get('county', 'N/A')}  
            **Crop:** {meta.get('crop', 'N/A')}  
            **Yield:** {meta.get('yield', 'N/A')}  
            **Latitude / Longitude:** {meta.get('Latitude', 'N/A')} / {meta.get('Longitude', 'N/A')}  
            **Other metadata:** { {k:v for k,v in meta.items() if k not in ['County','Crop','Yield','Latitude','Longitude']} }
            """)

# --- Step 5: How It Works ---
st.divider()
st.subheader("🛠 How This Works")

st.markdown("""
This dashboard uses **Retrieval-Augmented Generation (RAG)** to answer questions about corn yield data:

1. **User Query** : You type a question about the dataset.
2. **Embedding** : The question is converted into a numeric vector using Cohere embeddings.
3. **Vector Search** : Pinecone searches the dataset for the top relevant entries (top-k matches).
4. **Context Assembly** : Retrieved data is combined into a prompt for the AI.
5. **AI Answer Generation** : Cohere’s Chat model generates an answer using the retrieved context.
6. **Answer + Context Display** : The dashboard shows both the AI answer and the underlying data used.
""")

# Optional: Visual layout with columns for each step
st.markdown("### Visual Overview")
cols = st.columns(6)
steps = ["Query", "Embed", "Vector Search", "Context", "AI Answer", "Display"]
descriptions = [
    "User types a question",
    "Query is converted to vector",
    "Top-k relevant rows retrieved",
    "Data assembled into prompt",
    "AI generates answer",
    "Answer & context displayed"
]

for col, step, desc in zip(cols, steps, descriptions):
    col.markdown(f"**{step}**\n\n{desc}")

# --- Step 6a: Load CSV again (or reuse from ingestion) ---
csv_path = "corn_data.csv"
df = pd.read_csv(csv_path)

# Optional: convert coordinates to float
df['Latitude'] = df['Latitude'].astype(float)
df['Longitude'] = df['Longitude'].astype(float)
df['Yield'] = df['Yield'].astype(float)

# --- Step 6b: Metrics ---
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Farms", df['Farmer'].nunique())
col2.metric("Total Yield", df['Yield'].sum())
col3.metric("Average Acreage", round(df['Acreage'].mean(), 2))

st.subheader("🌍 Farm Locations Map")

# Use pydeck Deck with OpenStreetMap style
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v10',  # Mapbox light map
    initial_view_state=pdk.ViewState(
        latitude=df['Latitude'].mean(),   # Center on your data
        longitude=df['Longitude'].mean(),
        zoom=4,  # wider view for country/world
        pitch=0
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[Longitude, Latitude]',
            get_fill_color='[255, 140, 0, 160]',  # Orange for visibility
            get_radius='Yield * 100',  # scale by yield (adjust factor as needed)
            pickable=True
        )
    ],
    tooltip={
        "html": "<b>County:</b> {County} <br> <b>Crop:</b> {Crop} <br> <b>Yield:</b> {Yield}",
        "style": {"backgroundColor": "white", "color": "black"}
    }
))
