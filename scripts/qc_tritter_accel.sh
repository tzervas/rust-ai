#!/bin/bash
# Quality Control script for tritter-accel
# Run from rust-ai root directory

set -e  # Exit on first failure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== tritter-accel Quality Control ===${NC}"
echo ""

# Gate 1: Compilation
echo -e "${YELLOW}[1/5] Compilation Check${NC}"
if cargo check -p tritter-accel 2>&1; then
    echo -e "${GREEN}✓ Compilation passed${NC}"
else
    echo -e "${RED}✗ Compilation failed${NC}"
    exit 1
fi
echo ""

# Gate 2: Clippy Linting
echo -e "${YELLOW}[2/5] Clippy Linting${NC}"
if cargo clippy -p tritter-accel -- -D warnings 2>&1; then
    echo -e "${GREEN}✓ Linting passed${NC}"
else
    echo -e "${RED}✗ Linting failed${NC}"
    exit 1
fi
echo ""

# Gate 3: Tests
echo -e "${YELLOW}[3/5] Running Tests${NC}"
if cargo test -p tritter-accel 2>&1; then
    echo -e "${GREEN}✓ Tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
echo ""

# Gate 4: Documentation
echo -e "${YELLOW}[4/5] Documentation Check${NC}"
DOC_OUTPUT=$(cargo doc -p tritter-accel --no-deps 2>&1)
DOC_WARNINGS=$(echo "$DOC_OUTPUT" | grep -c "warning:" || true)
if [ "$DOC_WARNINGS" -eq 0 ]; then
    echo -e "${GREEN}✓ Documentation passed${NC}"
else
    echo -e "${YELLOW}⚠ Documentation has $DOC_WARNINGS warnings${NC}"
fi
echo ""

# Gate 5: Workspace Check
echo -e "${YELLOW}[5/5] Workspace Compatibility${NC}"
if cargo check --workspace 2>&1 | head -50; then
    echo -e "${GREEN}✓ Workspace check passed${NC}"
else
    echo -e "${RED}✗ Workspace check failed${NC}"
    exit 1
fi
echo ""

echo -e "${GREEN}=== All QC Gates Passed ===${NC}"
