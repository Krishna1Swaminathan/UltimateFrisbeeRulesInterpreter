"""
app.py — Flask backend for the Ultimate Frisbee Rules Interpreter.
Embeddings via HF Inference API (no torch, ~60MB RAM).
Runs on Render's free tier.
"""

import os
import json
import time

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
INDEX_NAME  = "usau-rules"
HF_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL  = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"
GROQ_MODEL  = "llama-3.3-70b-versatile"
TOP_K       = 6

app = Flask(__name__)
CORS(app, origins="*")

_pinecone_idx = None
_groq_client  = None


def get_index():
    global _pinecone_idx
    if _pinecone_idx is None:
        key = os.environ.get("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY not set.")
        _pinecone_idx = Pinecone(api_key=key).Index(INDEX_NAME)
    return _pinecone_idx


def get_groq():
    global _groq_client
    if _groq_client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set.")
        _groq_client = Groq(api_key=key)
    return _groq_client


def embed_query(text: str) -> list[float]:
    """Embed a single query string via HF Inference API."""
    hf_key  = os.environ.get("HF_API_KEY")
    if not hf_key:
        raise RuntimeError("HF_API_KEY not set.")

    headers = {"Authorization": f"Bearer {hf_key}"}

    for attempt in range(3):
        resp = requests.post(
            HF_API_URL,
            headers = headers,
            json    = {"inputs": text, "options": {"wait_for_model": True}},
            timeout = 30,
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data[0], list):
                return [sum(col) / len(col) for col in zip(*data)]
            return data
        elif resp.status_code == 503:
            wait = resp.json().get("estimated_time", 10)
            time.sleep(wait + 1)
        else:
            raise RuntimeError(f"HF API error {resp.status_code}: {resp.text}")

    raise RuntimeError("HF API unavailable after retries.")


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def retrieve(query: str) -> list[dict]:
    query_vec = embed_query(query)
    results   = get_index().query(
        vector           = query_vec,
        top_k            = TOP_K,
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
- Read every subsection in the retrieved text before ruling.
- Cite the most specific subsection number you can find.
- Be direct. Players want a clear ruling, not hedging.
- Keep explanation in plain language.
- Output ONLY the JSON object. No preamble, no markdown fences.
"""


def generate_ruling(scenario: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"[{c['title']}]\n{c['text']}" for c in chunks)
    user_message = f"GAME SCENARIO:\n{scenario}\n\nRETRIEVED RULE SECTIONS:\n{context}\n\nReturn your ruling as JSON."

    raw = get_groq().chat.completions.create(
        model       = GROQ_MODEL,
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature = 0.1,
        max_tokens  = 800,
    ).choices[0].message.content.strip()

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
        count = get_index().describe_index_stats().total_vector_count
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
        return jsonify({"error": "Scenario is too short."}), 400

    try:
        chunks = retrieve(scenario)
        ruling = generate_ruling(scenario, chunks)
        return jsonify(ruling)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "AI returned malformed response. Try again."}), 500
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🥏 Ultimate Rules Interpreter — port {port}")
    print(f"   Embedder: {HF_MODEL} via HF Inference API")
    print(f"   LLM:      {GROQ_MODEL} via Groq")
    app.run(host="0.0.0.0", port=port, debug=False)