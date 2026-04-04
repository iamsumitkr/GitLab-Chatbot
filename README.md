# GitLab GenAI Chatbot

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers user queries using GitLab’s Handbook and Direction pages.

## 🧠 Features

* Context-aware answers using RAG
* Source attribution for transparency
* Clean Streamlit UI
* Recursive data ingestion pipeline

## ⚙️ Tech Stack

* LangChain
* FAISS
* Gemini API
* Streamlit

## 📦 Setup Instructions

### 1. Clone Repo

git clone <your-repo-link>

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Add API Key

Create a `.env` file:
GOOGLE_API_KEY=your_api_key_here

### 4. Run Ingestion

python ingest.py

### 5. Run App

streamlit run app.py

## 🌍 Deployment

Deployed on Streamlit Cloud (link here)

## 📌 Architecture

User → Streamlit → Retriever → FAISS → Gemini → Response

## ✨ Improvements

* Metadata-based retrieval
* Deduplication in ingestion
* Guardrails to prevent hallucination
