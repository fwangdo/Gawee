#!/bin/bash
#===----------------------------------------------------------------------===//
# to_llvm_ir.sh - Convert Gawee MLIR to LLVM IR
#===----------------------------------------------------------------------===//
#
# Usage:
#   ./scripts/to_llvm_ir.sh <input.mlir> [output.ll]
#
# This script:
#   1. Runs gawee-opt with --scf-to-llvm to lower to LLVM dialect
#   2. Uses mlir-translate to convert LLVM dialect -> LLVM IR
#
# Example:
#   ./scripts/to_llvm_ir.sh test/llvm_test.mlir output.ll
#   llc output.ll -o output.s
#   clang output.s -o output
#
#===----------------------------------------------------------------------===//

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.mlir> [output.ll]"
    exit 1
fi

INPUT="$1"
OUTPUT="${2:-output.ll}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GAWEE_OPT="$PROJECT_DIR/build/gawee-opt"
MLIR_TRANSLATE="$HOME/llvm-install/bin/mlir-translate"

# Check tools exist
if [ ! -f "$GAWEE_OPT" ]; then
    echo "Error: gawee-opt not found at $GAWEE_OPT"
    echo "Run ./build.sh first"
    exit 1
fi

if [ ! -f "$MLIR_TRANSLATE" ]; then
    echo "Error: mlir-translate not found at $MLIR_TRANSLATE"
    exit 1
fi

echo "=== Step 1: Lower to LLVM dialect ==="
$GAWEE_OPT --scf-to-llvm "$INPUT" -o /tmp/llvm_dialect.mlir
echo "Output: /tmp/llvm_dialect.mlir"

echo ""
echo "=== Step 2: Translate to LLVM IR ==="
$MLIR_TRANSLATE --mlir-to-llvmir /tmp/llvm_dialect.mlir -o "$OUTPUT"
echo "Output: $OUTPUT"

echo ""
echo "=== Done! ==="
echo ""
echo "Next steps to compile to binary:"
echo "  llc $OUTPUT -o output.s -filetype=obj"
echo "  clang output.s -o output"
