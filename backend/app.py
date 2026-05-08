"""
app.py — Flask backend for the Ultimate Frisbee Rules Interpreter.

Endpoints:
    POST /api/interpret   — Main RAG pipeline: retrieve + generate
    GET  /api/health      — Quick liveness check
    GET  /api/status      — Check whether ChromaDB is populated

Run:
    GROQ_API_KEY=your_key python app.py
"""

import os
import json
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Config ───────────────────────────────────────────────────────────────────
CHROMA_PATH = Path(__file__).parent / "../data/chroma_db"
COLLECTION_NAME = "usau_rules"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"   # fast + capable; swap to llama3-8b-8192 if rate-limited
TOP_K = 6                          # rule sections to retrieve per query

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Lazy-load heavy objects once on first request
_embed_model = None
_chroma_collection = None
_groq_client = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma_collection = client.get_collection(COLLECTION_NAME)
    return _chroma_collection


def get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── RAG pipeline ─────────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    """Embed the query and pull the top-K matching rule sections."""
    model = get_embed_model()
    collection = get_collection()

    query_vec = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append(
            {
                "text": doc,
                "title": meta.get("title", "Unknown"),
                "section": meta.get("section", ""),
                "relevance_score": round(1 - dist, 3),  # cosine similarity approx
            }
        )
    return chunks


SYSTEM_PROMPT = """You are an expert Ultimate Frisbee rules interpreter with deep knowledge of
the USAU (USA Ultimate) 11th Edition rules. Your job is to analyze game scenarios and provide
clear, authoritative rulings grounded in the official rulebook.

You will be given:
1. A game scenario described by a player
2. Relevant rule sections retrieved from the USAU rulebook (each section may contain many subsections)

The rule sections use a lettered/numbered subsection format, e.g.:
  17. Violations and Fouls
  17.A. Traveling ...
  17.B. Pick: A pick occurs when ...
  17.B.1. A player may call "pick" ...

Read ALL subsections carefully — the answer is often in a subsection, not the heading.

Your response MUST be a valid JSON object with EXACTLY these fields:

{
  "ruling": "Travel | No Violation | Foul | Pick | Contest | Turnover | Out of Bounds | Other",
  "summary": "One sentence verdict (e.g. 'This is a valid pick call.')",
  "explanation": "2-4 sentences explaining WHY this ruling applies, citing the specific subsection (e.g. Rule 17.B.1). Use plain English.",
  "rule_reference": "The specific subsection(s) that apply (e.g. 'Rule 17.B — Pick').",
  "ambiguity_note": "Any genuine ambiguity or interpretation edge case. Empty string if none.",
  "retrieved_sections": []
}

Rules:
- Read every subsection in the retrieved text before ruling — do not stop at the heading.
- Cite the most specific subsection number you can find (e.g. 17.B.1 not just 17).
- Be direct. Players want a clear ruling, not hedging.
- Keep explanation in plain language.
- Output ONLY the JSON object. No preamble, no markdown fences.
"""


def generate_ruling(scenario: str, chunks: list[dict]) -> dict:
    """Send the scenario + retrieved chunks to Groq/Llama3 and parse the ruling."""
    context = "\n\n".join(
        f"[{c['title']}]\n{c['text']}" for c in chunks
    )

    user_message = f"""GAME SCENARIO:
{scenario}

RETRIEVED RULE SECTIONS:
{context}

Analyze this scenario using only the rule sections above and return your ruling as JSON."""

    client = get_groq()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,   # Low temp for consistent rulings
        max_tokens=800,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    ruling = json.loads(raw)

    # Attach the retrieved sections so the frontend can show them
    ruling["retrieved_sections"] = [
        {"title": c["title"], "text": c["text"], "relevance_score": c["relevance_score"]}
        for c in chunks
    ]

    return ruling


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/status", methods=["GET"])
def status():
    """Check if the vector DB is populated and ready."""
    try:
        collection = get_collection()
        count = collection.count()
        return jsonify({"ready": count > 0, "chunks": count})
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)}), 200


@app.route("/api/feedback", methods=["POST"])
def feedback():
    """Log a thumbs-up or thumbs-down on a ruling to feedback.jsonl."""
    data = request.get_json(silent=True)
    if not data or data.get("vote") not in ("up", "down"):
        return jsonify({"error": "Invalid feedback payload."}), 400

    log_path = Path(__file__).parent / "../data/feedback.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": data.get("timestamp"),
        "vote":      data.get("vote"),
        "scenario":  data.get("scenario", ""),
        "ruling":    data.get("ruling", ""),
    }

    with open(log_path, "a") as f:
        f.write(__import__("json").dumps(entry) + "\n")

    return jsonify({"status": "logged"})


@app.route("/api/interpret", methods=["POST"])
def interpret():
    data = request.get_json(silent=True)
    if not data or not data.get("scenario", "").strip():
        return jsonify({"error": "Please provide a scenario."}), 400

    scenario = data["scenario"].strip()
    if len(scenario) < 10:
        return jsonify({"error": "Scenario is too short. Describe what happened in more detail."}), 400

    try:
        chunks = retrieve(scenario)
        ruling = generate_ruling(scenario, chunks)
        return jsonify(ruling)

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "The AI returned a malformed response. Try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🥏 Ultimate Frisbee Rules Interpreter — Backend")
    print(f"   Chroma DB path : {CHROMA_PATH.resolve()}")
    print(f"   LLM model      : {GROQ_MODEL}")
    print(f"   Embedding model: {EMBED_MODEL}")
    print()

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY is not set. Set it before making requests.")

    app.run(debug=True, port=5000)
