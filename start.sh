#!/bin/bash
# start.sh — Quick start for the Ultimate Rules Interpreter
# Usage: ./start.sh

set -e

if [ -z "$GROQ_API_KEY" ]; then
  echo "❌ GROQ_API_KEY is not set."
  echo "   Get a free key at https://console.groq.com"
  echo "   Then run: export GROQ_API_KEY=your_key_here"
  exit 1
fi

if [ ! -f "data/usau_rules.pdf" ]; then
  echo "⚠️  data/usau_rules.pdf not found."
  echo "   Download from https://usaultimate.org/rules/"
  echo "   and save it as data/usau_rules.pdf"
  exit 1
fi

if [ ! -d "data/chroma_db" ]; then
  echo "📦 No ChromaDB found — running ingestion first..."
  cd backend
  python ingest.py --pdf ../data/usau_rules.pdf
  cd ..
  echo ""
fi

echo "🥏 Starting backend on http://localhost:5000"
echo "   Open frontend/index.html in your browser"
echo ""
cd backend && python app.py
