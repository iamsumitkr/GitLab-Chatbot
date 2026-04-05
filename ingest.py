import hashlib
import os
import re
import socket
import time
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

BASE_URL = "https://about.gitlab.com"
GEMINI_HOST = "generativelanguage.googleapis.com"
SEED_URLS = [
    "https://about.gitlab.com/handbook/",
    "https://about.gitlab.com/direction/",
    "https://about.gitlab.com/company/",
]
ALLOWED_PATH_PREFIXES = (
    "/handbook",
    "/direction",
    "/company",
)
SKIP_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".pdf",
    ".zip",
    ".mp4",
    ".webm",
)
MAX_DEPTH = 3
MAX_PAGES = int(os.getenv("MAX_CRAWL_PAGES", "30"))
MIN_TEXT_LENGTH = 200
REQUEST_TIMEOUT = 15
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "10"))
MAX_EMBED_RETRIES = int(os.getenv("MAX_EMBED_RETRIES", "6"))
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "90"))

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


def normalize_url(url):
    parsed = urlparse(url)
    clean_path = parsed.path.rstrip("/") or "/"

    normalized = parsed._replace(
        scheme="https",
        params="",
        query="",
        fragment="",
        path=clean_path,
    )
    return urlunparse(normalized)


def is_valid(url):
    parsed = urlparse(url)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.netloc == "about.gitlab.com"
        and parsed.path.startswith(ALLOWED_PATH_PREFIXES)
        and not parsed.path.endswith(SKIP_EXTENSIONS)
    )


def extract_page_text(soup):
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "noscript",
            "iframe",
            "svg",
            "form",
        ]
    ):
        tag.decompose()

    main_content = soup.find("main") or soup.find("article") or soup.body or soup

    sections = []

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title:
        sections.append(title)

    for heading in main_content.find_all(["h1", "h2", "h3"]):
        text = heading.get_text(" ", strip=True)
        if text:
            sections.append(text)

    body_text = main_content.get_text(separator=" ", strip=True)
    if body_text:
        sections.append(body_text)

    return "\n\n".join(sections)


def is_duplicate(text):
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if text_hash in seen_hashes:
        return True
    seen_hashes.add(text_hash)
    return False


def build_session():
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; GitLabChatbotIngest/1.0; "
                "+https://about.gitlab.com)"
            )
        }
    )
    return session


def get_retry_delay(message, default_delay=30):
    match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1)) + 1))
    return default_delay


def embed_documents_with_retries(embeddings_model, docs, batch_size=EMBED_BATCH_SIZE):
    text_embeddings = []
    total_docs = len(docs)

    for start in range(0, total_docs, batch_size):
        batch_docs = docs[start : start + batch_size]
        texts = [doc.page_content for doc in batch_docs]
        titles = [doc.metadata.get("title") or "" for doc in batch_docs]

        for attempt in range(1, MAX_EMBED_RETRIES + 1):
            try:
                print(
                    f"Embedding batch {start // batch_size + 1}/"
                    f"{(total_docs + batch_size - 1) // batch_size} "
                    f"({len(batch_docs)} chunks)"
                )
                vectors = embeddings_model.embed_documents(
                    texts,
                    batch_size=len(texts),
                    titles=titles,
                )
                text_embeddings.extend(zip(texts, vectors, strict=True))
                break
            except GoogleGenerativeAIError as exc:
                message = str(exc)
                if "RESOURCE_EXHAUSTED" not in message:
                    raise

                if attempt == MAX_EMBED_RETRIES:
                    raise SystemExit(
                        "Gemini embeddings quota was exhausted repeatedly. "
                        "Wait a minute and rerun, reduce MAX_CRAWL_PAGES, or upgrade your Gemini quota."
                    ) from exc

                delay = get_retry_delay(message)
                print(
                    f"Quota hit while embedding batch {start // batch_size + 1}. "
                    f"Retrying in {delay}s (attempt {attempt}/{MAX_EMBED_RETRIES})."
                )
                time.sleep(delay)

    return text_embeddings


def crawl(seed_urls, max_depth=MAX_DEPTH, max_pages=MAX_PAGES):
    session = build_session()
    queue = deque((normalize_url(url), 0) for url in seed_urls)

    while queue and len(visited) < max_pages:
        url, depth = queue.popleft()

        if url in visited or depth > max_depth:
            continue

        try:
            print(f"Scraping [{depth}/{max_depth}]: {url}")
            visited.add(url)

            res = session.get(url, timeout=REQUEST_TIMEOUT)
            res.raise_for_status()

            content_type = res.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            text = extract_page_text(soup)

            if len(text) < MIN_TEXT_LENGTH or is_duplicate(text):
                continue

            page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "title": page_title,
                        "depth": depth,
                    },
                )
            )

            if depth == max_depth:
                continue

            for link in soup.find_all("a", href=True):
                full_url = normalize_url(urljoin(url, link["href"]))
                if is_valid(full_url) and full_url not in visited:
                    queue.append((full_url, depth + 1))

        except requests.RequestException as exc:
            print(f"Request error for {url}: {exc}")
        except Exception as exc:
            print(f"Error processing {url}: {exc}")


crawl(SEED_URLS)

print(f"Total docs collected: {len(documents)}")

if not documents:
    raise SystemExit(
        "No documents were collected. Check the crawl settings or internet access and rerun."
    )

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=150,
)

chunked_docs = splitter.split_documents(documents)

if len(chunked_docs) > MAX_CHUNKS:
    print(f"Limiting chunks from {len(chunked_docs)} to {MAX_CHUNKS} to stay within quota.")
    chunked_docs = chunked_docs[:MAX_CHUNKS]

print(f"Total chunks: {len(chunked_docs)}")

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    task_type="retrieval_document",
)

ensure_google_api_key()
ensure_dns()

try:
    embedded_texts = embed_documents_with_retries(embeddings, chunked_docs)
    db = FAISS.from_embeddings(
        embedded_texts,
        embeddings,
        metadatas=[doc.metadata for doc in chunked_docs],
    )
    db.save_local("vectorstore")
except GoogleGenerativeAIError as exc:
    if "Temporary failure in name resolution" in str(exc):
        raise SystemExit(
            f"Google embeddings request failed because DNS lookup for {GEMINI_HOST} failed. "
            "Check your internet connection, VPN/proxy, or DNS settings and rerun."
        ) from exc
    raise

print("Optimized vector DB ready!")
