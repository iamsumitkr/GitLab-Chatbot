# GitLab GenAI Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers questions using content crawled from GitLab's public documentation pages, including Handbook, Direction, and Company pages.

Live app: https://gl-chatbot.streamlit.app/

## Features

- Streamlit chat interface for asking GitLab-related questions
- FAISS vector search over crawled GitLab content
- Gemini embeddings for retrieval and Gemini chat generation for answers
- Source snippets shown in the app for transparency
- Broader ingestion coverage with normalized links, bounded crawling, and deduplication
- Quota-aware embedding retries and configurable crawl/chunk limits

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini API
- BeautifulSoup

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/iamsumitkr/GitLab-Chatbot.git
cd GitLab-Chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env`

```env
GOOGLE_API_KEY=your_api_key_here
MAX_CRAWL_PAGES=30
EMBED_BATCH_SIZE=10
MAX_CHUNKS=90
```

### 4. Build the vector store

```bash
python ingest.py
```

### 5. Run the app

```bash
streamlit run app.py
```

## How It Works

1. `ingest.py` crawls selected GitLab pages and extracts clean text.
2. The text is split into chunks and embedded with Gemini.
3. FAISS stores the embeddings locally in `vectorstore/`.
4. `rag_chain.py` retrieves relevant chunks for each user query.
5. Gemini generates a grounded answer from the retrieved context.

## Deployment

Streamlit app: https://gl-chatbot.streamlit.app/

## Notes

- If Gemini embedding quota is exhausted, reduce `MAX_CRAWL_PAGES` or `MAX_CHUNKS` and rerun `python ingest.py`.
- `.env` is ignored by git and should not be committed.
