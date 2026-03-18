# RAG Research Chatbot

A chatbot that answers questions about research papers using RAG (Retrieval Augmented Generation).

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Start LM Studio server on localhost:1234
4. Provide a local dataset in `data/raw/nuai_publications.json` (a sample is already included).
5. Run: `python src/main.py`

## Usage
The chatbot will load research abstracts from the JSON file in `data/raw/` and answer questions based on them.
