#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_PATH="$ROOT_DIR/benchmarks/onnx/nlp/bert_tiny/onnx/model.onnx"
OUTPUT_PATH="$ROOT_DIR/artifacts/front_rewrite/bert_tiny.onnx"

cd "$ROOT_DIR"

.venv/bin/python3 -m front.onnx_rewrite.run_onnx_rewrite \
  --input "$INPUT_PATH" \
  --output "$OUTPUT_PATH"
