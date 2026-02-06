#!/bin/bash
# Script to dynamically patch Cargo.toml dependencies based on CI inputs
# This allows PRs to specify custom sister project versions via:
# 1. Workflow dispatch inputs
# 2. PR description markers (e.g., `peft-rs: feature-branch`)
# 3. Environment variables

set -e

PEFT_REF="${PEFT_RS_REF:-main}"
QLORA_REF="${QLORA_RS_REF:-main}"
UNSLOTH_REF="${UNSLOTH_RS_REF:-main}"

# Check if PR description contains dependency overrides
# Format: `peft-rs: branch-name` or `qlora-rs: v1.0.0` or `unsloth-rs: commit-sha`
if [ -n "$PR_BODY" ]; then
    echo "Checking PR description for dependency overrides..."
    
    # Extract peft-rs ref if specified
    if echo "$PR_BODY" | grep -q "peft-rs:"; then
        PEFT_REF=$(echo "$PR_BODY" | grep -oP "peft-rs:\s*\K\S+" | head -1)
        echo "  Found peft-rs override: $PEFT_REF"
    fi
    
    # Extract qlora-rs ref if specified
    if echo "$PR_BODY" | grep -q "qlora-rs:"; then
        QLORA_REF=$(echo "$PR_BODY" | grep -oP "qlora-rs:\s*\K\S+" | head -1)
        echo "  Found qlora-rs override: $QLORA_REF"
    fi
    
    # Extract unsloth-rs ref if specified
    if echo "$PR_BODY" | grep -q "unsloth-rs:"; then
        UNSLOTH_REF=$(echo "$PR_BODY" | grep -oP "unsloth-rs:\s*\K\S+" | head -1)
        echo "  Found unsloth-rs override: $UNSLOTH_REF"
    fi
fi

echo ""
echo "Patching Cargo.toml with sister project refs:"
echo "  peft-rs:    $PEFT_REF"
echo "  qlora-rs:   $QLORA_REF"
echo "  unsloth-rs: $UNSLOTH_REF"
echo ""

# Create a temporary sed script
cat > /tmp/patch.sed <<EOF
s|git = "https://github.com/tzervas/peft-rs", branch = "[^"]*"|git = "https://github.com/tzervas/peft-rs", branch = "$PEFT_REF"|g
s|git = "https://github.com/tzervas/qlora-rs", branch = "[^"]*"|git = "https://github.com/tzervas/qlora-rs", branch = "$QLORA_REF"|g
s|git = "https://github.com/tzervas/unsloth-rs", branch = "[^"]*"|git = "https://github.com/tzervas/unsloth-rs", branch = "$UNSLOTH_REF"|g
EOF

# Apply patches to Cargo.toml
sed -i -f /tmp/patch.sed Cargo.toml

echo "Cargo.toml patched successfully!"
echo ""
echo "Sister project dependencies in Cargo.toml:"
grep -A 1 "github.com/tzervas/peft-rs\|github.com/tzervas/qlora-rs\|github.com/tzervas/unsloth-rs" Cargo.toml || true
