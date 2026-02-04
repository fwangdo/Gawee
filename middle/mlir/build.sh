#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_DIR="$HOME/llvm-install"
TBLGEN="$LLVM_DIR/bin/mlir-tblgen"

echo "=== Step 1: Generate TableGen files ==="

mkdir -p include/Gawee/generated

$TBLGEN --gen-dialect-decls \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.h.inc

$TBLGEN --gen-dialect-defs \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.cpp.inc

$TBLGEN --gen-op-decls \
  -I $LLVM_DIR/include \
  -I include \
  -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.h.inc

$TBLGEN --gen-op-defs \
  -I $LLVM_DIR/include \
  -I include \
  -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.cpp.inc

# Workaround: Fix broken string literals in generated code (MLIR TableGen bug)
sed -i '' 's/"requires attribute/" "requires attribute/g' include/Gawee/generated/GaweeOps.cpp.inc

echo "Generated files:"
ls -la include/Gawee/generated/

echo ""
echo "=== Step 2: Configure CMake ==="

rm -rf build
mkdir build
cd build

cmake .. -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Symlink compile_commands.json to project root for IDE support
ln -sf "$SCRIPT_DIR/build/compile_commands.json" "$SCRIPT_DIR/compile_commands.json"

echo ""
echo "=== Step 3: Build ==="

make

echo ""
echo "=== Done! ==="
