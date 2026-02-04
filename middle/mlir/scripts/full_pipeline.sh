#!/bin/bash
# Full Gawee compilation pipeline
# Usage: ./scripts/full_pipeline.sh input.mlir

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLVM_DIR="$HOME/llvm-install"

INPUT="$1"

if [ -z "$INPUT" ]; then
  echo "Usage: $0 <input.mlir>"
  exit 1
fi

echo "=== Step 1: Gawee -> Linalg ==="
STEP1=$("$PROJECT_DIR/build/gawee-opt" --convert-gawee-to-linalg "$INPUT")
echo "$STEP1"

echo ""
echo "=== Step 2: Bufferize (tensor -> memref) ==="
STEP2=$(echo "$STEP1" | "$LLVM_DIR/bin/mlir-opt" --one-shot-bufferize="bufferize-function-boundaries")
echo "$STEP2"

echo ""
echo "=== Step 3: Linalg -> SCF Loops ==="
STEP3=$(echo "$STEP2" | "$LLVM_DIR/bin/mlir-opt" --convert-linalg-to-loops)
echo "$STEP3"

echo ""
echo "=== Done! ==="
