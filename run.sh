#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
SPAMSCOPE_PYTHON="${SPAMSCOPE_PYTHON:-python3}"
SPAMSCOPE_PORT="${SPAMSCOPE_PORT:-8765}"

if [ ! -d .venv ]; then
  "$SPAMSCOPE_PYTHON" -m venv .venv
fi

if ! .venv/bin/python -c "import streamlit, sklearn, torch, sentence_transformers" >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements.txt
fi

exec .venv/bin/python -m streamlit run app.py --server.port "$SPAMSCOPE_PORT"
