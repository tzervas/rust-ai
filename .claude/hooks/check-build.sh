#!/usr/bin/env bash
# Hook: Stop
# Verifies that the workspace still compiles after Claude finishes a response.
# Only runs if Rust files were likely modified (checks git diff).
# Non-blocking informational check.

set -euo pipefail

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

# Check if any Rust files were modified in the working tree
CHANGED_RS=$(git diff --name-only --diff-filter=M 2>/dev/null | grep '\.rs$' || true)
STAGED_RS=$(git diff --cached --name-only --diff-filter=M 2>/dev/null | grep '\.rs$' || true)

if [[ -z "$CHANGED_RS" && -z "$STAGED_RS" ]]; then
  exit 0
fi

# Quick compile check
if ! cargo check --workspace 2>/dev/null; then
  echo "Warning: workspace has compilation errors after recent changes."
  exit 0
fi

exit 0
