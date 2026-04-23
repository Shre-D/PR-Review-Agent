#!/bin/bash
CHANGED=$(git status --short | wc -l | tr -d ' ')
if [ "$CHANGED" -gt 0 ]; then
  echo ""
  echo "── Session ended: $CHANGED uncommitted files ──"
  echo "Commit with: git add -A && git commit -m 'checkpoint'"
fi
