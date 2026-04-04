import hashlib
import os
import socket

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urljoin, urlparse

load_dotenv()

BASE_URL = "https://about.gitlab.com"
GEMINI_HOST = "generativelanguage.googleapis.com"

visited = set()
documents = []
seen_hashes = set()


def ensure_google_api_key():
    if os.getenv("GOOGLE_API_KEY"):
        return

    raise SystemExit(
        "Missing GOOGLE_API_KEY in your environment. Add it to .env and rerun."
    )


def ensure_dns():
    try:
        socket.gethostbyname(GEMINI_HOST)
    except socket.gaierror as exc:
        raise SystemExit(
            f"Cannot resolve {GEMINI_HOST}. Check your internet connection or DNS settings and try again."
        ) from exc

# -------------------------------
# Helper: Clean HTML
# -------------------------------
def clean_html(soup):
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator=" ", strip=True)

# -------------------------------
# Helper: Deduplication
# -------------------------------
def is_duplicate(text):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in seen_hashes:
        return True
    seen_hashes.add(text_hash)
    return False

# -------------------------------
# URL Filter
# -------------------------------
def is_valid(url):
    parsed = urlparse(url)
    return (
        parsed.netloc == "about.gitlab.com"
        and ("/handbook" in url or "/direction" in url)
        and "#" not in url
    )

# -------------------------------
# Crawl Function
# -------------------------------
def crawl(url, depth=2):
    if url in visited or depth == 0:
        return
    
    try:
        print(f"Scraping: {url}")
        visited.add(url)

        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        text = clean_html(soup)

        if len(text) < 200:
            return

        if is_duplicate(text):
            return

        # Store as Document with metadata
        documents.append(
            Document(
                page_content=text,
                metadata={"source": url}
            )
        )

        # Find next links
        for link in soup.find_all("a", href=True):
            full_url = urljoin(BASE_URL, link["href"])

            if is_valid(full_url):
                crawl(full_url, depth - 1)

    except Exception as e:
        print(f"Error: {e}")

# -------------------------------
# Start Crawling
# -------------------------------
crawl("https://about.gitlab.com/handbook/", depth=2)
crawl("https://about.gitlab.com/direction/", depth=2)

print(f"Total docs collected: {len(documents)}")

# -------------------------------
# Chunking (SMART)
# -------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunked_docs = splitter.split_documents(documents)

print(f"Total chunks: {len(chunked_docs)}")

# -------------------------------
# Embeddings
# -------------------------------
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document",
)

# -------------------------------
# Vector Store
# -------------------------------
ensure_google_api_key()
ensure_dns()

try:
    db = FAISS.from_documents(chunked_docs, embeddings)
    db.save_local("vectorstore")
except GoogleGenerativeAIError as exc:
    if "Temporary failure in name resolution" in str(exc):
        raise SystemExit(
            f"Google embeddings request failed because DNS lookup for {GEMINI_HOST} failed. "
            "Check your internet connection, VPN/proxy, or DNS settings and rerun."
        ) from exc
    raise

print("✅ Optimized vector DB ready!")
