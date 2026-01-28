import streamlit as st
import cohere
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import pandas as pd
import pydeck as pdk

load_dotenv()

# Initialize clients
co = cohere.Client(os.getenv("COHERE_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Page config
st.set_page_config(
    page_title="🌾 Agricultural Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main background with modern gradient - Deep Navy to Teal */
    .main {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling - Modern Dark with Cyan accents */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .sidebar-content {
        color: white;
        padding: 1rem;
    }
    
    .sidebar-title {
        color: #00d4ff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .sidebar-section {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #00d4ff;
    }
    
    .sidebar-section h3 {
        color: #00d4ff;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-section p {
        color: #e0e0e0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .sidebar-stat {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    .sidebar-stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .sidebar-stat-label {
        font-size: 0.8rem;
        color: #b0b0b0;
        text-transform: uppercase;
    }
    
    /* Header styling - Modern Gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-subtitle {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-top: 4px solid;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Answer card - Modern Gradient */
    .answer-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }
    
    .answer-title {
        color: white;
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .answer-text {
        color: white;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    
    /* Context cards */
    .context-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #00d4ff;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Step indicator */
    .step-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        height: 100%;
        border-top: 3px solid #00d4ff;
    }
    
    .step-title {
        font-weight: 700;
        color: #1a1a2e;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .step-desc {
        font-size: 0.85rem;
        color: #666;
    }
    
    /* Query input styling */
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d4ff;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.1);
    }
    
    /* Button styling - Modern Cyan/Pink gradient */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 212, 255, 0.3);
        width: 100%;
        height: 52px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.5);
    }
    
    /* Section headers */
    .section-header {
        color: #1a1a2e;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #00d4ff;
        display: inline-block;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Fix input alignment */
    .stTextInput > label {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ====================== Sidebar ======================
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌾 Dashboard Menu</div>', unsafe_allow_html=True)
    
    # About Section
    st.markdown("""
    <div class="sidebar-section">
        <h3>📊 About This Dashboard</h3>
        <p>An AI-powered agricultural analytics platform that helps you explore corn yield data using natural language queries.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    csv_path = "corn_data.csv"
    df = pd.read_csv(csv_path)
    df['Latitude'] = df['Latitude'].astype(float)
    df['Longitude'] = df['Longitude'].astype(float)
    df['Yield'] = df['Yield'].astype(float)
    
    st.markdown("""
    <div class="sidebar-section">
        <h3>📈 Quick Statistics</h3>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{df.shape[0]}</div>
            <div class="sidebar-stat-label">Total Records</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{df['County'].nunique()}</div>
            <div class="sidebar-stat-label">Unique Counties</div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-value">{df['Yield'].max():,.0f}</div>
            <div class="sidebar-stat-label">Max Yield</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="sidebar-section">
        <h3>✨ Key Features</h3>
        <p>• Natural language search<br>
        • AI-powered insights<br>
        • Interactive map visualization<br>
        • Real-time data retrieval<br>
        • Context-aware answers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Technology Stack
    st.markdown("""
    <div class="sidebar-section">
        <h3>🛠️ Tech Stack</h3>
        <p>• Cohere AI<br>
        • Pinecone Vector DB<br>
        • Streamlit<br>
        • PyDeck Maps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Example Queries
    st.markdown("""
    <div class="sidebar-section">
        <h3>💡 Example Queries</h3>
        <p style="font-size: 0.85rem;">
        • "Which counties have the highest yield?"<br>
        • "Show me farms with low acreage"<br>
        • "What's the average yield per county?"<br>
        • "List all crops in the dataset"
        </p>
    </div>
    """, unsafe_allow_html=True)

# ====================== Header ======================
st.markdown("""
<div class="main-header">
    <div class="main-title">🌾 Agricultural Intelligence Dashboard</div>
    <div class="main-subtitle">Unlock insights from corn yield data with AI-powered natural language search</div>
</div>
""", unsafe_allow_html=True)

# ====================== Metrics ======================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #667eea;">
        <div style="font-size: 2.5rem;">👨‍🌾</div>
        <div class="metric-value" style="color: #667eea;">{df['Farmer'].nunique()}</div>
        <div class="metric-label">Total Farms</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #00d4ff;">
        <div style="font-size: 2.5rem;">🌾</div>
        <div class="metric-value" style="color: #00d4ff;">{df['Yield'].sum():,.0f}</div>
        <div class="metric-label">Total Yield (bushels)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card" style="border-top-color: #f5576c;">
        <div style="font-size: 2.5rem;">📏</div>
        <div class="metric-value" style="color: #f5576c;">{df['Acreage'].mean():.1f}</div>
        <div class="metric-label">Average Acreage</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ====================== User Query ======================
st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col_query, col_button = st.columns([5, 1])
with col_query:
    query = st.text_input(
        "query",
        placeholder="e.g., Which counties have the highest corn yield?",
        label_visibility="collapsed",
        key="query_input"
    )
with col_button:
    ask_button = st.button("🔍 Search", use_container_width=True, key="search_btn")

# ====================== AI Retrieval & Answer ======================
if ask_button and query:
    with st.spinner("🔄 Analyzing your question..."):
        # Step 1: Embed query
        response = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"]
        )
        query_embedding = response.embeddings.float[0]

        # Step 2: Query Pinecone
        # Retrieve all available records for comprehensive analysis
        results = index.query(
            vector=query_embedding,
            top_k=1000,
            include_metadata=True
        )

        # Step 3: Assemble context + Smart sorting based on query type
        context_texts = []
        
        # Detect query type to determine sorting strategy
        query_lower = query.lower()
        yield_keywords = ['yield', 'harvest', 'production', 'bushels', 'most corn', 'highest production']
        acreage_keywords = ['acreage', 'acres', 'land', 'farm size']
        fertilizer_keywords = ['fertilizer', 'chemicals', 'inputs']
        
        is_yield_query = any(keyword in query_lower for keyword in yield_keywords)
        is_acreage_query = any(keyword in query_lower for keyword in acreage_keywords)
        is_fertilizer_query = any(keyword in query_lower for keyword in fertilizer_keywords)
        
        # Sort matches based on query type
        if is_yield_query:
            sorted_matches = sorted(
                results['matches'],
                key=lambda x: float(x['metadata'].get('yield', 0)),
                reverse=True
            )
            sort_type = "yield"
        elif is_acreage_query:
            sorted_matches = sorted(
                results['matches'],
                key=lambda x: float(x['metadata'].get('acreage', 0)),
                reverse=True
            )
            sort_type = "acreage"
        elif is_fertilizer_query:
            sorted_matches = sorted(
                results['matches'],
                key=lambda x: float(x['metadata'].get('fertilizer_amount', 0)),
                reverse=True
            )
            sort_type = "fertilizer"
        else:
            # Default: sort by semantic relevance score
            sorted_matches = results['matches']
            sort_type = "relevance"
        
        for match in sorted_matches:
            meta = match['metadata']
            context_texts.append(
                f"Farmer: {meta.get('farmer', 'Unknown')}, "
                f"County: {meta.get('county', '')}, "
                f"Crop: {meta.get('crop', '')}, "
                f"Yield: {meta.get('yield', '')} bushels, "
                f"Acreage: {meta.get('acreage', '')} acres, "
                f"Education: {meta.get('education', '')}, "
                f"Gender: {meta.get('gender', '')}, "
                f"Age: {meta.get('age_bracket', '')}, "
                f"Fertilizer: {meta.get('fertilizer_amount', '')}, "
                f"Laborers: {meta.get('laborers', '')}, "
                f"Water Source: {meta.get('water_source', '')}, "
                f"Power Source: {meta.get('power_source', '')}"
            )
        context_block = "\n".join(context_texts)

        # Step 4: Generate answer
        prompt = f"""You are an agricultural data expert. Using the following farmer data, answer the user's question accurately and comprehensively.

IMPORTANT INSTRUCTIONS:
- Always look through ALL the data provided to find complete answers
- When asked "who has the highest/most X", identify the farmer with the maximum value
- When asked to list farmers, provide ALL farmer names from the data that match the criteria
- For aggregation questions (highest, most, best), compare ALL values in the data
- ALWAYS cite specific farmer names and their values from the data
- If comparing yields, acreage, fertilizer, or any metric - analyze across all provided records
- Format farmer names clearly when listing multiple farmers

Retrieved Farm Data (all relevant records):
{context_block}

User Question: {query}

Answer:"""
        chat_response = co.chat(
            model="command-a-03-2025",
            message=prompt,
            temperature=0
        )
        answer_text = chat_response.text.strip()

    # ====================== Display Answer ======================
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-title">🤖 AI-Generated Answer</div>
        <div class="answer-text">{answer_text}</div>
    </div>
    """, unsafe_allow_html=True)

    # ====================== Display Retrieved Context ======================
    st.markdown('<div class="section-header">📄 Retrieved Context</div>', unsafe_allow_html=True)
    
    # Display sort type to user
    sort_labels = {
        "yield": "🌾 Top 5 by Yield",
        "acreage": "📏 Top 5 by Acreage",
        "fertilizer": "⚗️ Top 5 by Fertilizer Amount",
        "relevance": "🎯 Top 5 Most Relevant"
    }
    sort_label = sort_labels.get(sort_type, "Top 5 Results")
    st.markdown(f"<p style='color: #666; margin-bottom: 1rem;'>{sort_label} from {len(sorted_matches)} matching records.</p>", unsafe_allow_html=True)
    
    for i, match in enumerate(sorted_matches[:5], start=1):
        meta = match['metadata']
        
        # Create dynamic header based on sort type
        if sort_type == "yield":
            header_text = f"📍 #{i} — {meta.get('farmer', 'Unknown')} | Yield: {meta.get('yield', 'N/A')} bushels"
        elif sort_type == "acreage":
            header_text = f"📍 #{i} — {meta.get('farmer', 'Unknown')} | Acreage: {meta.get('acreage', 'N/A')} acres"
        elif sort_type == "fertilizer":
            header_text = f"📍 #{i} — {meta.get('farmer', 'Unknown')} | Fertilizer: {meta.get('fertilizer_amount', 'N/A')} units"
        else:
            header_text = f"📍 Context #{i} — Relevance Score: {match.get('score', 0):.3f}"
        
        with st.expander(header_text, expanded=(i==1)):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown(f"**Farmer:** {meta.get('farmer', 'N/A')}")
                st.markdown(f"**County:** {meta.get('county', 'N/A')}")
                st.markdown(f"**Crop:** {meta.get('crop', 'N/A')}")
                st.markdown(f"**Yield:** {meta.get('yield', 'N/A')} bushels")
                st.markdown(f"**Acreage:** {meta.get('acreage', 'N/A')} acres")
            with col_b:
                st.markdown(f"**Gender:** {meta.get('gender', 'N/A')}")
                st.markdown(f"**Age:** {meta.get('age_bracket', 'N/A')}")
                st.markdown(f"**Education:** {meta.get('education', 'N/A')}")
                st.markdown(f"**Household Size:** {meta.get('household_size', 'N/A')}")
                st.markdown(f"**Laborers:** {meta.get('laborers', 'N/A')}")
            with col_c:
                st.markdown(f"**Fertilizer:** {meta.get('fertilizer_amount', 'N/A')}")
                st.markdown(f"**Water Source:** {meta.get('water_source', 'N/A')}")
                st.markdown(f"**Power Source:** {meta.get('power_source', 'N/A')}")
                st.markdown(f"**Advisory Source:** {meta.get('advisory_source', 'N/A')}")
                st.markdown(f"**Crop Insurance:** {meta.get('crop_insurance', 'N/A')}")

# ====================== Farm Map ======================
st.markdown('<div class="section-header">🗺️ Farm Locations Map</div>', unsafe_allow_html=True)
st.markdown("<p style='color: #666; margin-bottom: 1rem;'>Interactive map showing all farm locations. Circle size represents yield volume.</p>", unsafe_allow_html=True)

st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=df['Latitude'].mean(),
        longitude=df['Longitude'].mean(),
        zoom=4,
        pitch=0
    ),
    layers=[
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[Longitude, Latitude]',
            get_fill_color='[0, 212, 255, 200]',
            get_radius='Yield * 100',
            pickable=True
        )
    ],
    tooltip={
        "html": "<b>County:</b> {County}<br><b>Crop:</b> {Crop}<br><b>Yield:</b> {Yield} bushels",
        "style": {"backgroundColor": "#1a1a2e", "color": "#00d4ff", "fontSize": "14px"}
    }
))

# ====================== How It Works ======================
st.markdown('<div class="section-header">🛠️ How This Works</div>', unsafe_allow_html=True)

st.markdown("""
<p style='color: #666; font-size: 1.05rem; line-height: 1.7; margin-bottom: 2rem;'>
This dashboard leverages <b>Retrieval-Augmented Generation (RAG)</b> to provide intelligent answers about agricultural data. 
Here's the process:
</p>
""", unsafe_allow_html=True)

cols = st.columns(6)
steps = [
    ("Query", "User types a natural language question", "💬"),
    ("Embed", "Question converted to vector embedding", "🔢"),
    ("Search", "Top-k relevant records retrieved", "🔍"),
    ("Context", "Data assembled into structured prompt", "📋"),
    ("Generate", "AI produces contextual answer", "🤖"),
    ("Display", "Answer & sources presented", "✨")
]

for col, (step, desc, emoji) in zip(cols, steps):
    col.markdown(f"""
    <div class="step-box">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{emoji}</div>
        <div class="step-title">{step}</div>
        <div class="step-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

# ====================== Footer ======================
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; padding: 2rem 0; border-top: 1px solid #e0e0e0; margin-top: 3rem;'>
    <p>Powered by <b>Cohere</b> + <b>Pinecone</b> | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)