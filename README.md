# 🥏 Ultimate Frisbee Rules Interpreter

A RAG-powered rules assistant for USAU Ultimate Frisbee.
Describe a game scenario in plain English → get a structured ruling with
explanation, rule reference, and ambiguity notes — grounded in the official rulebook.

## Architecture

```
User scenario
     ↓
Sentence-transformer embedding (all-MiniLM-L6-v2)
     ↓
ChromaDB vector retrieval (top-4 rule sections)
     ↓
Groq API — Llama 3 70B (structured JSON output)
     ↓
Ruling + Explanation + Rule Reference + Ambiguity Note
```

## Tech Stack

| Layer       | Tool                                   |
|-------------|----------------------------------------|
| LLM         | Llama 3 70B via Groq API (free tier)  |
| Embeddings  | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB   | ChromaDB (local, persistent)           |
| Backend     | Flask + Flask-CORS                     |
| Frontend    | Plain HTML/CSS/JS (no framework)       |
| PDF parsing | pypdf                                  |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

1. Go to https://console.groq.com
2. Sign up for a free account
3. Create an API key
4. Set it as an environment variable:

```bash
export GROQ_API_KEY=your_key_here
```

### 3. Get the USAU rulebook PDF

Download the official USAU 11th Edition rules PDF from:
https://usaultimate.org/rules/

Save it as: `data/usau_rules.pdf`

### 4. Ingest the rulebook (run once)

```bash
cd backend
python ingest.py --pdf ../data/usau_rules.pdf
```

This will:
- Extract text from the PDF
- Chunk by rule section
- Embed each chunk with sentence-transformers
- Store everything in ChromaDB at `data/chroma_db/`

You'll see output like:
```
📄 Reading PDF: ../data/usau_rules.pdf
   Extracted 87,234 characters
✂️  Chunking by rule section…
   142 chunks created
🧠 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
📦 Connecting to ChromaDB…
⚙️  Embedding and storing chunks…
✅ Done! 142 rule sections stored in ChromaDB.
```

### 5. Start the Flask backend

```bash
cd backend
python app.py
```

Server starts at http://localhost:5000

### 6. Open the frontend

Open `frontend/index.html` in your browser.
(No web server needed — just open the file directly.)

## How to use

1. Type or paste a game scenario in the text box
2. Click **Get Ruling** (or press Cmd/Ctrl+Enter)
3. See the structured ruling:
   - **Ruling** — Travel / Foul / No Violation / etc.
   - **Explanation** — Plain English reasoning
   - **Rule Reference** — Exact USAU section that applies
   - **Ambiguity Note** — If the scenario is genuinely unclear
4. Click "Show retrieved rule sections" to see what the RAG pipeline retrieved

## Project Structure

```
ultimate-rules-interpreter/
├── backend/
│   ├── app.py          Flask API server
│   └── ingest.py       PDF → ChromaDB ingestion pipeline
├── frontend/
│   └── index.html      Single-page web UI
├── data/
│   ├── usau_rules.pdf  ← You add this
│   └── chroma_db/      ← Created by ingest.py
├── requirements.txt
└── README.md
```

## Key concepts (for presentation)

### RAG (Retrieval-Augmented Generation)
Instead of relying solely on Llama 3's training data, we:
1. Store the actual rulebook as searchable vector embeddings
2. At query time, retrieve the most relevant sections
3. Feed those sections to the LLM as grounding context

This means the LLM can't hallucinate rules — it's working from the actual text.

### Why structured output?
The system prompt forces JSON output with specific fields.
This makes the ruling machine-readable and lets the UI render each
component (ruling, explanation, reference) separately.

### Chunking strategy
The rulebook is split at rule section boundaries (e.g. "11.3.2 ...").
This is better than fixed-size windows because rule sections are
naturally semantic units — each chunk has a clear meaning.

## Troubleshooting

**"DB not populated"** — Run ingest.py first.

**"Backend offline"** — Make sure app.py is running on port 5000.

**"GROQ_API_KEY not set"** — Export the env variable before starting app.py.

**Chunking produces 0 chunks** — The PDF may use a different heading format.
ingest.py will fall back to window chunking automatically, but you may want
to inspect the extracted text and adjust the regex in `chunk_by_section()`.

**Rate limited by Groq** — Switch `GROQ_MODEL` in app.py from `llama3-70b-8192`
to `llama3-8b-8192` (faster, smaller, still very capable).
