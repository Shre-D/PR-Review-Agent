#!/bin/bash
# Auto-syntax-check any Python file that was just written or edited
for f in $(git diff --name-only 2>/dev/null | grep '\.py$'); do
  if python3 -m py_compile "$f" 2>/dev/null; then
    echo "  ✓ syntax OK: $f"
  else
    echo "  ✗ SYNTAX ERROR: $f"
    python3 -m py_compile "$f" 2>&1 | head -3
  fi
done
