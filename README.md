# 🌾 Agricultural Intelligence Dashboard

An AI-powered platform that transforms agricultural data exploration through natural language queries. Ask questions about corn yield data and get intelligent, context-aware answers backed by retrieval-augmented generation (RAG).

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Features

- **🗣️ Natural Language Queries** - Ask questions in plain English about your agricultural data
- **🤖 AI-Powered Answers** - Leverages Cohere's advanced language models for intelligent responses
- **⚡ Vector Search** - Fast semantic search powered by Pinecone vector database
- **🗺️ Interactive Maps** - Visualize farm locations with yield data on an interactive PyDeck map
- **📊 Real-time Analytics** - View quick statistics and insights at a glance
- **📄 Source Context** - Every answer includes the retrieved data used to generate it
- **🎨 Modern UI** - Beautiful, responsive interface with gradient themes and smooth interactions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI/LLM** | Cohere API (embed-english-v3.0, command-a-03-2025) |
| **Vector DB** | Pinecone |
| **Visualization** | PyDeck (Mapbox) |
| **Data Processing** | Pandas |
| **Language** | Python 3.8+ |

---

## 📋 How It Works

The dashboard implements **Retrieval-Augmented Generation (RAG)** in 6 steps:

```
1. Query Input → 2. Vector Embedding → 3. Semantic Search → 
4. Context Assembly → 5. AI Generation → 6. Display Results
```

**Process Flow:**
1. User enters a natural language question
2. Question is converted to a vector embedding using Cohere
3. Top-K relevant records are retrieved from Pinecone
4. Retrieved data is assembled into a structured prompt
5. Cohere's language model generates a contextual answer
6. Answer and source data are displayed to the user

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- API Keys for:
  - [Cohere AI](https://cohere.com/)
  - [Pinecone](https://www.pinecone.io/)

### Installation

1. **Clone the repository**
```bash
git clone git@github.com:Pratham-Gauswami/cohere-ag-rag.git
cd cohere-ag-rag
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the root directory:
```env
COHERE_API_KEY=your_cohere_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

5. **Prepare your data**
- Ensure `corn_data.csv` is in the root directory
- CSV should contain columns: `County`, `Crop`, `Yield`, `Acreage`, `Farmer`, `Latitude`, `Longitude`

6. **Run the application**
```bash
streamlit run app3.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📁 Project Structure

```
cohere-ag-rag/
├── app3.py                 # Main Streamlit application
├── csv_ingest.py          # Data ingestion and preparation
├── corn_data.csv          # Agricultural dataset
├── .env                   # Environment variables (git ignored)
├── .gitignore             # Git ignore configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── previousVersionFiles/  # Legacy versions
│   ├── app.py
│   ├── app2.py
│   └── final_RAGline.py
└── testFiles/             # Testing and validation
    ├── test_embedding.py
    ├── test_pinecone.py
    └── vectorTest.py
```

---

## 💡 Example Queries

Try asking questions like:

- "Which counties have the highest corn yield?"
- "Show me farms with low acreage"
- "What's the average yield per county?"
- "List all crops in the dataset"
- "Which farms have the best yield-to-acreage ratio?"
- "Compare yields across different counties"

---

## 📊 Data Format

Your CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `County` | String | County name |
| `Crop` | String | Crop type |
| `Yield` | Float | Yield in bushels |
| `Acreage` | Float | Farm acreage |
| `Farmer` | String | Farmer name |
| `Latitude` | Float | Geographic latitude |
| `Longitude` | Float | Geographic longitude |

---

## 🔑 Environment Variables

```env
# Cohere API Key for embeddings and chat
COHERE_API_KEY=your_key_here

# Pinecone configuration
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

**Security Note:** Never commit your `.env` file to version control. It's included in `.gitignore` by default.

---

## 🎨 UI Features

- **Responsive Layout** - Adapts to different screen sizes
- **Dark Theme** - Modern gradient backgrounds with cyan accents
- **Interactive Elements** - Expandable cards, hover effects, smooth transitions
- **Real-time Statistics** - Live metric cards showing key data points
- **Tooltip Support** - Hover over map markers for details
- **Error Handling** - Graceful error messages and feedback

---

## 🐛 Troubleshooting

### Map not showing?
- Ensure your DataFrame has `Latitude` and `Longitude` columns
- Check that coordinates are valid (latitude: -90 to 90, longitude: -180 to 180)

### No results from queries?
- Verify your Cohere API key is valid
- Check that Pinecone index is properly populated
- Ensure your data has been properly ingested into Pinecone

### Import errors?
- Reinstall dependencies: `pip install -r requirements.txt`
- Verify you're using the correct Python version (3.8+)

---

## 📝 Requirements

```
streamlit==1.28.0
cohere==4.0.0
pinecone-client==3.0.0
python-dotenv==1.0.0
pandas==2.0.0
pydeck==0.8.0
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙋 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code comments
3. Open an issue on GitHub

---

## 🌟 Acknowledgments

Built with:
- [Cohere](https://cohere.com/) - Advanced AI & language models
- [Pinecone](https://www.pinecone.io/) - Vector database
- [Streamlit](https://streamlit.io/) - Web app framework
- [PyDeck](https://deckgl.readthedocs.io/en/latest/) - Map visualization

---

**Made with ❤️ for agricultural intelligence**
