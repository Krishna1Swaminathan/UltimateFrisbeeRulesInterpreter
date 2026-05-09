"""
ingest.py — Parse the USAU rulebook PDF, chunk by TOP-LEVEL rule section,
embed with sentence-transformers, and store in Pinecone.

Run this ONCE locally before deploying to Render.
Your vectors live in Pinecone permanently — no need to re-run on redeploy.

Usage:
    PINECONE_API_KEY=your_key python ingest.py --pdf ../data/usau_rules.pdf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_NAME      = "usau-rules"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM       = 384        # dimension for all-MiniLM-L6-v2
MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2000
BATCH_SIZE      = 50         # Pinecone upsert batch size


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_by_section(text: str) -> list[dict]:
    """
    Chunk by TOP-LEVEL section only (e.g. Rule 1, Rule 11, Rule 17),
    keeping all subsections together so retrieval has full context.
    """
    top_level_pattern = re.compile(
        r"(?m)^(\d{1,2})\.\s+([A-Z][^\n]{2,80})\n"
    )
    matches = list(top_level_pattern.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_id    = match.group(1).strip()
        section_title = match.group(2).strip()
        body          = text[start:end].strip()

        if len(body) > MAX_CHUNK_CHARS:
            for j, sub in enumerate(split_long_chunk(body, MAX_CHUNK_CHARS)):
                global_idx = len(chunks)
                headed = f"Rule {section_id}. {section_title}\n\n{sub}"
                chunks.append({
                    "id":      f"chunk_{global_idx}_{section_id}_part{j}",
                    "title":   f"Rule {section_id}. {section_title} (part {j+1})",
                    "text":    headed,
                    "section": section_id,
                })
        elif len(body) >= MIN_CHUNK_CHARS:
            global_idx = len(chunks)
            chunks.append({
                "id":      f"chunk_{global_idx}_{section_id}",
                "title":   f"Rule {section_id}. {section_title}",
                "text":    body,
                "section": section_id,
            })

    # Fallback: window chunking
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


# ── Pinecone ingestion ────────────────────────────────────────────────────────
def ingest(pdf_path: str):
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set.")
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

    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("🌲 Connecting to Pinecone…")
    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    existing = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME in existing:
        print(f"   Deleting existing index '{INDEX_NAME}'…")
        pc.delete_index(INDEX_NAME)
        time.sleep(5)

    print(f"   Creating index '{INDEX_NAME}' (dim={EMBED_DIM}, metric=cosine)…")
    pc.create_index(
        name   = INDEX_NAME,
        dimension = EMBED_DIM,
        metric = "cosine",
        spec   = ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    # Wait for index to be ready
    print("   Waiting for index to be ready…")
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("   Index ready.")

    index = pc.Index(INDEX_NAME)

    print("⚙️  Embedding and upserting chunks…")
    texts      = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Batch upsert
    vectors = [
        (c["id"], emb, {"title": c["title"], "section": c["section"], "text": c["text"]})
        for c, emb in zip(chunks, embeddings)
    ]

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i : i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"   Upserted batch {i // BATCH_SIZE + 1}/{-(-len(vectors) // BATCH_SIZE)}")

    print(f"\n✅ Done! {len(chunks)} rule sections stored in Pinecone index '{INDEX_NAME}'.")
    print(f"   These vectors persist in the cloud — no need to re-run on redeploy.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest USAU rulebook PDF into Pinecone")
    parser.add_argument("--pdf", required=True, help="Path to USAU rules PDF")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"❌ File not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf)