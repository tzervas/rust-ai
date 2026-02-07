#!/usr/bin/env bash
# Hook: PostToolUse (Write|Edit)
# Runs cargo fmt check on modified Rust files to catch formatting issues early.
# Non-blocking: exit 0 always, feedback via stdout.

set -euo pipefail

INPUT=$(cat)
TOOL_INPUT=$(echo "$INPUT" | jq -r '.tool_input // empty' 2>/dev/null)
FILE_PATH=$(echo "$TOOL_INPUT" | jq -r '.file_path // .path // empty' 2>/dev/null)

# Only lint Rust files
if [[ "$FILE_PATH" != *.rs ]]; then
  exit 0
fi

# Find the crate root by looking for Cargo.toml
DIR=$(dirname "$FILE_PATH")
while [[ "$DIR" != "/" && "$DIR" != "." ]]; do
  if [[ -f "$DIR/Cargo.toml" ]]; then
    CRATE_ROOT="$DIR"
    break
  fi
  DIR=$(dirname "$DIR")
done

if [[ -z "${CRATE_ROOT:-}" ]]; then
  exit 0
fi

# Run fmt check (non-blocking)
FMT_OUTPUT=$(cargo fmt --manifest-path "$CRATE_ROOT/Cargo.toml" --check 2>&1) || true

if [[ -n "$FMT_OUTPUT" ]]; then
  echo '{"decision": "block", "reason": "Rust formatting issues detected. Run cargo fmt."}'
  exit 2
fi

exit 0
