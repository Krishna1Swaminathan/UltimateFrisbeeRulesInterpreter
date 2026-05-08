"""
ingest.py — Parse the USAU rulebook PDF, chunk by TOP-LEVEL rule section,
embed with sentence-transformers, and store in ChromaDB.

Usage:
    python ingest.py --pdf ../data/usau_rules.pdf

Run this ONCE before starting the server. Re-run whenever
the rulebook PDF changes (delete data/chroma_db first).
"""

import argparse
import re
import sys
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent / "../data/chroma_db"
COLLECTION_NAME = "usau_rules"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MIN_CHUNK_CHARS = 80
MAX_CHUNK_CHARS = 2000   # Larger cap so full rule sections stay together


# ── PDF extraction ───────────────────────────────────────────────────────────
def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_by_section(text: str) -> list[dict]:
    """
    Chunk by TOP-LEVEL section only (e.g. '3.', '11.', '16.'),
    keeping all subsections (3.A, 3.A.1, 3.A.2 ...) together in one chunk.

    This prevents orphaned fragments — e.g. the pick rule (17.B) stays
    with its siblings (17.A, 17.C) so retrieval gets the full context.
    """
    # Match ONLY top-level sections: single integer + period + uppercase title
    top_level_pattern = re.compile(
        r"(?m)^(\d{1,2})\.\s+([A-Z][^\n]{2,80})\n"
    )

    matches = list(top_level_pattern.finditer(text))
    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_id = match.group(1).strip()
        section_title = match.group(2).strip()
        body = text[start:end].strip()

        if len(body) > MAX_CHUNK_CHARS:
            sub_chunks = split_long_chunk(body, MAX_CHUNK_CHARS)
            for j, sub in enumerate(sub_chunks):
                global_idx = len(chunks)
                # Prepend heading to every sub-chunk so model has context
                headed = f"Rule {section_id}. {section_title}\n\n{sub}"
                chunks.append({
                    "id": f"chunk_{global_idx}_{section_id}_part{j}",
                    "title": f"Rule {section_id}. {section_title} (part {j+1})",
                    "text": headed,
                    "section": section_id,
                })
        elif len(body) >= MIN_CHUNK_CHARS:
            global_idx = len(chunks)
            chunks.append({
                "id": f"chunk_{global_idx}_{section_id}",
                "title": f"Rule {section_id}. {section_title}",
                "text": body,
                "section": section_id,
            })

    # Fallback: try subsection-level pattern
    if not chunks:
        print("⚠️  Top-level pattern matched nothing — trying subsection pattern.")
        sub_pattern = re.compile(
            r"(?m)^(\d+(?:\.\d+)*)[\.\s]+([A-Z][^\n]{0,80})\n"
        )
        matches = list(sub_pattern.finditer(text))
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_id = match.group(1).strip()
            section_title = match.group(2).strip()
            body = text[start:end].strip()
            if len(body) >= MIN_CHUNK_CHARS:
                global_idx = len(chunks)
                chunks.append({
                    "id": f"chunk_{global_idx}_{section_id}",
                    "title": f"{section_id} {section_title}",
                    "text": body,
                    "section": section_id,
                })

    if not chunks:
        print("⚠️  No sections found — falling back to window chunking.")
        chunks = window_chunk(text)

    return chunks


def split_long_chunk(text: str, max_len: int) -> list[str]:
    """Break a long string at paragraph boundaries."""
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
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append({
            "id": f"chunk_{idx}",
            "title": f"Section chunk {idx}",
            "text": text[start:end],
            "section": str(idx),
        })
        start += size - overlap
        idx += 1
    return chunks


# ── Embedding + storage ───────────────────────────────────────────────────────
def ingest(pdf_path: str):
    print(f"📄 Reading PDF: {pdf_path}")
    text = extract_text(pdf_path)
    print(f"   Extracted {len(text):,} characters")

    print("✂️  Chunking by top-level rule section…")
    chunks = chunk_by_section(text)
    print(f"   {len(chunks)} chunks created")

    # Print first few so you can verify they look right
    print("\n   Preview of first 5 chunks:")
    for c in chunks[:5]:
        print(f"   [{c['id']}] {c['title'][:60]}  ({len(c['text'])} chars)")
    print()

    print(f"🧠 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print("📦 Connecting to ChromaDB…")
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    try:
        client.delete_collection(COLLECTION_NAME)
        print("   Dropped existing collection")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    print("⚙️  Embedding and storing chunks…")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"title": c["title"], "section": c["section"]} for c in chunks],
    )

    print(f"\n✅ Done! {len(chunks)} rule sections stored in ChromaDB.")
    print(f"   Database path: {CHROMA_PATH.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest USAU rulebook PDF into ChromaDB")
    parser.add_argument("--pdf", required=True, help="Path to USAU rules PDF")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"❌ File not found: {args.pdf}")
        sys.exit(1)

    ingest(args.pdf)
