# 🥏 Ultimate Frisbee Rules Interpreter

A RAG-powered rules assistant for USAU Ultimate Frisbee.
Describe a game scenario in plain English → get a structured ruling with
explanation, rule reference, and ambiguity notes — grounded in the official rulebook.

🌐 **Live demo:** https://ultimatefrisbeerulesinterpreter.onrender.com

---

## How it works

```
User scenario
     ↓
HF Inference API — all-MiniLM-L6-v2 embedding (no local model)
     ↓
Pinecone vector retrieval (top-6 rule sections)
     ↓
Groq API — Llama 3.3 70B (structured JSON output)
     ↓
Ruling + Explanation + Rule Reference + Ambiguity Note
```

## Tech Stack

| Layer       | Tool                                              |
|-------------|---------------------------------------------------|
| LLM         | Llama 3.3 70B Versatile via Groq API (free tier) |
| Embeddings  | all-MiniLM-L6-v2 via HF Inference API (no torch) |
| Vector DB   | Pinecone Serverless (cloud, free tier)            |
| Backend     | Flask + Flask-CORS (serves frontend + API)        |
| Frontend    | Vanilla HTML/CSS/JS (no framework, no build step) |
| PDF parsing | pypdf                                             |
| Hosting     | Render (free tier)                                |

---

## Local Development Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/ultimate-rules-interpreter.git
cd ultimate-rules-interpreter
pip install -r requirements.txt
```

### 2. Get API keys (all free, no credit card required)

| Service | Where to get it |
|---------|----------------|
| Groq | https://console.groq.com |
| Pinecone | https://app.pinecone.io |
| Hugging Face | https://huggingface.co/settings/tokens |

```bash
export GROQ_API_KEY=your_groq_key
export PINECONE_API_KEY=your_pinecone_key
export HF_API_KEY=your_hf_token
```

### 3. Get the USAU rulebook PDF

Download the official USAU 11th Edition rules from:
https://usaultimate.org/rules/

Save it as: `data/usau_rules.pdf`

### 4. Ingest the rulebook (run once)

This embeds the rulebook and uploads vectors to Pinecone.
You only need to run this once — vectors persist in the cloud.

```bash
cd backend
python ingest.py --pdf ../data/usau_rules.pdf
```

Expected output:
```
📄 Reading PDF: ../data/usau_rules.pdf
   Extracted 87,234 characters
✂️  Chunking by top-level rule section…
   22 chunks created
🚀 Embedding via HF Inference API...
   Embedded 22/22
🌲 Setting up Pinecone...
   Index ready.
✅ Done! 22 rule sections stored in Pinecone.
```

### 5. Start the backend

```bash
cd backend
python app.py
```

Server starts at http://localhost:5000
Open http://localhost:5000 in your browser — Flask serves the frontend directly.

---

## Deployment (Render)

1. Push the repo to GitHub
2. Go to https://render.com → New Web Service → connect your repo
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python backend/app.py`
   - **Instance type:** Free
4. Add environment variables:
   - `GROQ_API_KEY`
   - `PINECONE_API_KEY`
   - `HF_API_KEY`
5. Deploy — Render gives you a live URL

> **Note:** Free tier Render services sleep after 15 minutes of inactivity.
> The first request after sleep may take ~10 seconds to respond.

---

## Project Structure

```
ultimate-rules-interpreter/
├── backend/
│   ├── app.py          Flask API + frontend server
│   └── ingest.py       PDF → Pinecone ingestion pipeline
├── frontend/
│   └── index.html      Single-page web UI (served by Flask)
├── data/
│   └── usau_rules.pdf  ← You add this (not committed to git)
├── Procfile            Render start command
├── runtime.txt         Python version for Render
├── requirements.txt
├── README.md
└── CLAUDE.md           AI collaboration context
```

---

## How to use

1. Type or paste a game scenario in natural language
2. Click **Get Ruling** (or press Cmd/Ctrl+Enter)
3. See the structured ruling:
   - **Ruling** — Travel / Foul / Pick / No Violation / etc.
   - **Explanation** — Plain English reasoning with subsection citations
   - **Rule Reference** — Exact USAU section that applies
   - **Ambiguity Note** — Flags genuine edge cases
4. Click "Show retrieved rule sections" to inspect the RAG context
5. Use the history sidebar to revisit previous rulings in the session

---

## Key Concepts

### RAG (Retrieval-Augmented Generation)
Instead of relying on an LLM's training data (which may hallucinate rules),
the system stores the actual USAU rulebook as vector embeddings and retrieves
the most relevant sections at query time. The LLM reasons over real rule text.

### Chunking strategy
The rulebook is chunked at **top-level rule boundaries only** (Rule 1, Rule 11,
Rule 17, etc.), keeping all subsections (17.A, 17.B.1, etc.) together in one
chunk. This prevents orphaned fragments and ensures the retriever always returns
semantically complete rule sections.

### API-first embeddings
`sentence-transformers` runs locally but requires ~500MB RAM — too much for
Render's free tier. Instead, the HF Inference API embeds text remotely.
The server makes an HTTP call and gets back a vector, using ~60MB RAM total.

---

## Troubleshooting

**HF API 404** — The endpoint URL may have changed. Current correct URL:
`https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction`

**Pinecone index not found** — Run `ingest.py` first to populate the index.

**Groq rate limited** — Switch `GROQ_MODEL` in `app.py` to `llama-3.1-8b-instant`
for higher rate limits at slightly lower quality.

**Chunking produces 0 chunks** — The PDF may use non-standard heading formatting.
`ingest.py` falls back to window chunking automatically. Inspect the extracted
text and adjust the regex in `chunk_by_section()` if needed.
