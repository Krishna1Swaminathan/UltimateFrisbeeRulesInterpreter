"""
ingest.py — Parse the USAU rulebook PDF, chunk by TOP-LEVEL rule section,
embed via Hugging Face Inference API (all-MiniLM-L6-v2), store in Pinecone.

No payment method required. Free HF Inference API key at:
https://huggingface.co/settings/tokens

Run ONCE locally before deploying to Render.

Usage:
    PINECONE_API_KEY=... HF_API_KEY=... python ingest.py --pdf ../data/usau_rules.pdf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import requests
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME      = "usau-rules"
HF_MODEL        = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL      = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
EMBED_DIM       = 384
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2000
PINECONE_BATCH  = 50


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(
        page.extract_text() for page in reader.pages if page.extract_text()
    )


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_by_section(text: str) -> list[dict]:
    pattern = re.compile(r"(?m)^(\d{1,2})\.\s+([A-Z][^\n]{2,80})\n")
    matches = list(pattern.finditer(text))
    chunks  = []

    for i, match in enumerate(matches):
        start         = match.start()
        end           = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_id    = match.group(1).strip()
        section_title = match.group(2).strip()
        body          = text[start:end].strip()

        if len(body) > MAX_CHUNK_CHARS:
            for j, sub in enumerate(split_long_chunk(body, MAX_CHUNK_CHARS)):
                chunks.append({
                    "id":      f"chunk_{len(chunks)}_{section_id}_part{j}",
                    "title":   f"Rule {section_id}. {section_title} (part {j+1})",
                    "text":    f"Rule {section_id}. {section_title}\n\n{sub}",
                    "section": section_id,
                })
        elif len(body) >= MIN_CHUNK_CHARS:
            chunks.append({
                "id":      f"chunk_{len(chunks)}_{section_id}",
                "title":   f"Rule {section_id}. {section_title}",
                "text":    body,
                "section": section_id,
            })

    if not chunks:
        print("⚠️  No top-level sections found — falling back to window chunking.")
        chunks = window_chunk(text)

    return chunks


def split_long_chunk(text: str, max_len: int) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text)
    sub_chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_len and current:
            sub_chunks.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        sub_chunks.append(current.strip())
    return sub_chunks


def window_chunk(text: str, size: int = 800, overlap: int = 100) -> list[dict]:
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({
            "id":      f"chunk_{idx}",
            "title":   f"Section chunk {idx}",
            "text":    text[start:end],
            "section": str(idx),
        })
        start += size - overlap
        idx   += 1
    return chunks


# ── Embedding via HF Inference API ────────────────────────────────────────────
def embed_texts(texts: list[str], hf_key: str) -> list[list[float]]:
    """
    Call HF Inference API one chunk at a time.
    The free tier has rate limits so we retry on 503 (model loading).
    """
    headers    = {"Authorization": f"Bearer {hf_key}"}
    embeddings = []

    for i, text in enumerate(texts):
        while True:
            resp = requests.post(
                HF_API_URL,
                headers = headers,
                json    = {"inputs": text, "options": {"wait_for_model": True}},
                timeout = 30,
            )

            if resp.status_code == 200:
                data = resp.json()
                # HF returns either a list of floats or a list-of-lists (mean pool)
                if isinstance(data[0], list):
                    # token-level → mean pool
                    vec = [sum(col) / len(col) for col in zip(*data)]
                else:
                    vec = data
                embeddings.append(vec)
                print(f"   Embedded {i+1}/{len(texts)}", end="\r")
                break

            elif resp.status_code == 503:
                wait = resp.json().get("estimated_time", 10)
                print(f"   Model loading, waiting {wait:.0f}s…")
                time.sleep(wait + 1)

            else:
                print(f"\n❌ HF API error {resp.status_code}: {resp.text}")
                sys.exit(1)

    print()  # newline after \r progress
    return embeddings


# ── Main ingestion ────────────────────────────────────────────────────────────
def ingest(pdf_path: str):
    hf_key       = os.environ.get("HF_API_KEY")
    pinecone_key = os.environ.get("PINECONE_API_KEY")

    if not hf_key:
        print("❌ HF_API_KEY not set.")
        print("   Get a free token at https://huggingface.co/settings/tokens")
        sys.exit(1)
    if not pinecone_key:
        print("❌ PINECONE_API_KEY not set.")
        sys.exit(1)

    print(f"📄 Reading PDF: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"   Extracted {len(text):,} characters")

    print("✂️  Chunking by top-level rule section…")
    chunks = chunk_by_section(text)
    print(f"   {len(chunks)} chunks created")

    print("\n   Preview of first 5 chunks:")
    for c in chunks[:5]:
        print(f"   [{c['id']}] {c['title'][:60]}  ({len(c['text'])} chars)")
    print()

    print(f"🚀 Embedding via HF Inference API ({HF_MODEL})…")
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts, hf_key)

    print("\n🌲 Setting up Pinecone…")
    pc       = Pinecone(api_key=pinecone_key)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        print(f"   Deleting existing index '{INDEX_NAME}'…")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)

    print(f"   Creating index '{INDEX_NAME}' (dim={EMBED_DIM}, cosine)…")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = EMBED_DIM,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    print("   Waiting for index to be ready…")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("   Index ready.")

    index = pc.Index(INDEX_NAME)

    print("⚙️  Upserting to Pinecone…")
    vectors = [
        (c["id"], emb, {"title": c["title"], "section": c["section"], "text": c["text"]})
        for c, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), PINECONE_BATCH):
        batch = vectors[i : i + PINECONE_BATCH]
        index.upsert(vectors=batch)
        print(f"   Upserted batch {i // PINECONE_BATCH + 1}/{-(-len(vectors) // PINECONE_BATCH)}")

    print(f"\n✅ Done! {len(chunks)} rule sections in Pinecone ('{INDEX_NAME}').")
