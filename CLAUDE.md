# CLAUDE.md — AI Collaboration Context

This file documents how Claude (Anthropic) was used to build the
Ultimate Frisbee Rules Interpreter as part of CSCI 455/555: Generative AI
for Software Development, Spring 2026.

---

## Project Summary

A RAG-powered web application that accepts plain-English descriptions of
Ultimate Frisbee game scenarios and returns structured rulings grounded in
the official USAU 11th Edition rulebook. Built entirely through iterative
conversation with Claude Sonnet via claude.ai.

---

## How Claude Was Used

This project was built using a "vibe coding" workflow: every component was
generated through natural language prompts, run locally, debugged by pasting
errors back into the conversation, and iterated on. No code was written from
scratch without AI assistance.

### What Claude generated

- Full RAG pipeline architecture (ingest → embed → retrieve → generate)
- `backend/ingest.py` — PDF parsing, chunking, embedding, Pinecone upsert
- `backend/app.py` — Flask API with all routes, RAG logic, prompt engineering
- `frontend/index.html` — Complete UI including layout, CSS, pipeline visualizer,
  history sidebar, result rendering, and example scenarios
- All deployment configuration (Procfile, runtime.txt)
- This README and CLAUDE.md

### Prompting approach

Prompts were consistently structured as either:
1. **Feature requests** — "Add a history sidebar that shows previous rulings,
   color-coded by ruling type, with relative timestamps and click-to-restore"
2. **Error reports** — Pasting the full traceback and describing what was
   expected vs. what happened

The most effective prompts were specific and included concrete evidence:
- ✅ "The pick call scenario returns equipment rules — here is what the
  retrieved sections show. I think the issue is the chunks are too small."
- ❌ "The answers aren't great, can you improve them?"

---

## Key Iteration Cycles

### 1. Initial architecture
**Prompt:** Described the full project — RAG pipeline for USAU rulebook,
natural language input, structured ruling output with verdict/explanation/
citation, Flask backend, plain HTML frontend.
**Result:** Working prototype in one pass. ChromaDB + sentence-transformers
+ Groq, full frontend with pipeline visualizer.

### 2. ChromaDB duplicate ID bug
**Prompt:** Pasted the full `DuplicateIDError` traceback.
**Diagnosis:** Multiple rule sections shared the same short numeric ID
(e.g. section "9" appeared in multiple parts of the PDF).
**Fix:** Prepend a global index counter to every chunk ID.

### 3. Poor retrieval quality
**Prompt:** "The pick call scenario returns equipment rules. I think the
rulebook has lots of subsections and that might be confusing the model."
**Root cause:** Chunking at every subsection heading produced 140+ tiny
fragments. Section 17.B.1 was separated from 17.B (the pick definition),
so retrieval found orphaned subsections with no context.
**Fix:** Chunk only at top-level rule numbers (Rule 1, Rule 11, Rule 17),
keeping all subsections together. Increased MAX_CHUNK_CHARS to 2000.
Also updated system prompt to explicitly tell the model about the
subsection format and instruct it to read all subsections before ruling.

### 4. CORS / frontend connection issue
**Symptom:** "Could not reach backend" despite Flask running correctly.
**Attempted fixes:** Flask-CORS config changes, Python http.server — neither worked.
**Root cause:** Opening index.html via file:// triggers browser security
policies that block fetch() requests regardless of CORS headers.
**Fix:** Changed BACKEND URL from `localhost` to `127.0.0.1` (browsers
treat these differently for CORS), and ultimately served the frontend
directly from Flask to eliminate the issue entirely.

### 5. Deployment — Out of Memory on Render
**Symptom:** App crashed on Render free tier (512MB RAM) before handling
any requests.
**Root cause:** `sentence-transformers` + `torch` load a ~500MB model into
RAM at startup, leaving no room for the OS and Python runtime.
**Fix:** Replaced local model with HF Inference API — the server makes an
HTTP call to embed text instead of loading the model locally. RAM usage
dropped from ~650MB to ~60MB.

### 6. Voyage AI rate limits
**Attempted:** Switched to Voyage AI for embeddings (better quality).
**Problem:** Free tier without a credit card is capped at 3 RPM — too slow
for ingestion and unreliable for production queries.
**Fix:** Switched to HF Inference API hosting the same all-MiniLM-L6-v2
model — free, no card required, same 384-dimension vectors.

### 7. HF API URL change
**Symptom:** 404 error on every embedding request after deployment.
**Root cause:** HF migrated their inference infrastructure. The old
`api-inference.huggingface.co/pipeline/...` path no longer works.
**Fix:** Updated to the new router URL:
`https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction`

### 8. Flask not serving frontend (404 on `/`)
**Symptom:** Render deployment showed "service live" but `/` returned 404.
**Root cause:** Flask only had `/api/*` routes — no route for the root path.
**Fix:** Added `send_from_directory` routes for `/` and `/<filename>`.
Set `BACKEND = ''` in index.html so API calls use the same origin
automatically (works on both localhost and Render with no changes).

---

## Where Claude Excelled

- **Boilerplate elimination** — Flask setup, Pinecone integration, CORS
  config, PDF parsing all generated correctly on the first attempt
- **Debugging from tracebacks** — Pasting an error almost always produced
  an immediate correct diagnosis and fix
- **Frontend polish** — The pipeline visualizer, color-coded badges,
  collapsible RAG context drawer, and history sidebar were all generated
  from plain English descriptions
- **Architecture decisions** — Correctly recommended Groq over HF Inference
  for the LLM (faster, more reliable structured output), explained the
  tradeoffs clearly

## Where Claude Struggled

- **Retrieval quality required domain knowledge** — The chunking bug was
  invisible from reading the code; it only manifested at runtime with real
  queries. Required the developer to understand why retrieval was failing
  before Claude could fix it.
- **CORS debugging** — Took three iterations. The actual root cause
  (file:// vs http:// origin) was only identified through independent research.
- **Artifact vs. disk divergence** — One iteration generated a revised
  frontend in a preview but never wrote it to disk. The running app stayed
  on the old version until this was caught.
- **External API instability** — HF's URL migration and Voyage's rate limit
  restrictions were runtime failures that couldn't be anticipated from code
  review alone.

---

## Model and Tools Used

- **Model:** Claude Sonnet (claude.ai web interface)
- **Session type:** Single extended conversation, not separate sessions
- **Other AI tools:** None — no Copilot, no other LLMs

---

## Prompting Patterns That Worked Best

```
Pattern 1 — Error report:
"I'm getting this error: [full traceback]
Expected behavior: [what should happen]
Actual behavior: [what is happening]"

Pattern 2 — Feature spec:
"Add [feature]. It should [behavior 1], [behavior 2].
When the user does X, it should Y."

Pattern 3 — Diagnosis request:
"[Symptom]. Here is the relevant output: [evidence].
What do you think is causing this and how do we fix it?"
```

The worst prompts were vague: "make it better", "it's not working".
The best prompts had concrete evidence attached.

---

## Files Entirely AI-Generated

- `backend/app.py`
- `backend/ingest.py`
- `frontend/index.html`
- `requirements.txt`
- `Procfile`
- `runtime.txt`
- `README.md`
- `CLAUDE.md` (this file)

The developer's role was: problem framing, requirement specification,
runtime testing, debugging diagnosis, and deployment decisions.
