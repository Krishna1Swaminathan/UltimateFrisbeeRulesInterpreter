"""
app.py — Flask backend for the Ultimate Frisbee Rules Interpreter.
Uses Pinecone for vector storage (works on Render's free tier).

Endpoints:
    POST /api/interpret   — RAG pipeline: retrieve + generate
    GET  /api/health      — Liveness check
    GET  /api/status      — Check Pinecone index is populated

Run locally:
    GROQ_API_KEY=... PINECONE_API_KEY=... python app.py
"""

import os
import json
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# Add this import at the top
from flask import Flask, request, jsonify, send_from_directory

# ... (keep your existing setup code)



# ... (keep your /api/ routes below)

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME  = "usau-rules"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.3-70b-versatile"
TOP_K       = 6

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")


# Add this route BEFORE your other routes
@app.route("/")
def serve_frontend():
    return send_from_directory("frontend", "index.html")


_embed_model   = None
_pinecone_idx  = None
_groq_client   = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_index():
    global _pinecone_idx
    if _pinecone_idx is None:
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY environment variable not set.")
        pc = Pinecone(api_key=api_key)
        _pinecone_idx = pc.Index(INDEX_NAME)
    return _pinecone_idx


def get_groq():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    model = get_embed_model()
    index = get_index()

    query_vec = model.encode([query]).tolist()[0]
    results   = index.query(
        vector          = query_vec,
        top_k           = TOP_K,
        include_metadata = True,
    )

    chunks = []
    for match in results.matches:
        meta = match.metadata or {}
        chunks.append({
            "text":            meta.get("text", ""),
            "title":           meta.get("title", "Unknown"),
            "section":         meta.get("section", ""),
            "relevance_score": round(match.score, 3),
        })
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
    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)

    user_message = f"""GAME SCENARIO:
{scenario}

RETRIEVED RULE SECTIONS:
{context}

Analyze this scenario using only the rule sections above and return your ruling as JSON."""

    client   = get_groq()
    response = client.chat.completions.create(
        model    = GROQ_MODEL,
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0.1,
        max_tokens  = 800,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if model adds them
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    ruling = json.loads(raw)
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
    try:
        index = get_index()
        stats = index.describe_index_stats()
        count = stats.total_vector_count
        return jsonify({"ready": count > 0, "chunks": count})
    except Exception as e:
        return jsonify({"ready": False, "error": str(e)}), 200


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
    port = int(os.environ.get("PORT", 5000))
    print(f"🥏 Ultimate Frisbee Rules Interpreter — Backend (port {port})")
    print(f"   LLM:       {GROQ_MODEL}")
    print(f"   Embeddings:{EMBED_MODEL}")
    print(f"   Pinecone:  index '{INDEX_NAME}'")
    print()

    if not os.environ.get("GROQ_API_KEY"):
        print("⚠️  WARNING: GROQ_API_KEY is not set.")
    if not os.environ.get("PINECONE_API_KEY"):
        print("⚠️  WARNING: PINECONE_API_KEY is not set.")

    app.run(host="0.0.0.0", port=port, debug=False)