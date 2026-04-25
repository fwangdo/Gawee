#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_DIR="$HOME/llvm-install"
ONNX_INCLUDE_DIR="$SCRIPT_DIR/../../.venv/lib/python3.12/site-packages"
PROTOBUF_DIR="/opt/homebrew/Cellar/protobuf@21/21.12/lib/cmake/protobuf"
PROTOC="$SCRIPT_DIR/../../.venv/lib/python3.12/site-packages/torch/bin/protoc"

echo "=== Step 1: Configure CMake ==="
# CMake now regenerates the TableGen .inc files automatically whenever
# GaweeDialect.td or GaweeOps.td changes.

rm -rf build
mkdir build
cd build

cmake .. \
  -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" \
  -DGAWEE_ONNX_INCLUDE_DIR="$ONNX_INCLUDE_DIR" \
  -DProtobuf_DIR="$PROTOBUF_DIR" \
  -DProtobuf_PROTOC_EXECUTABLE="$PROTOC" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Symlink compile_commands.json to project root for IDE support
ln -sf "$SCRIPT_DIR/build/compile_commands.json" "$SCRIPT_DIR/compile_commands.json"

echo ""
echo "=== Step 2: Build ==="
# The build now regenerates TableGen outputs before compiling.

make

echo ""
echo "=== Done! ==="
