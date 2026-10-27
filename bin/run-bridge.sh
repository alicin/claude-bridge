#!/usr/bin/env bash
# Strips inherited Claude Code session vars (e.g. CLAUDE_CODE_SIMPLE forces API-key-only auth)
# so the nested CLI uses this user's normal OAuth login.
set -euo pipefail
cd "$(dirname "$0")/.."
for var in $(env | cut -d= -f1 | grep -E '^(CLAUDE|ANTHROPIC)' || true); do unset "$var"; done
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
exec "${PYTHON:-.venv/bin/python}" -c 'from src.main import run_server; run_server()'
