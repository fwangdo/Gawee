#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LLVM_DIR="$HOME/llvm-install"
TBLGEN="$LLVM_DIR/bin/mlir-tblgen"

echo "=== Step 1: Generate TableGen files ==="
# mlir-tblgen reads .td files and generates C++ fragments (.inc files).
# .inc files are NOT standalone -- they are #included inside your .cpp/.h files.

mkdir -p include/Gawee/generated

# --gen-dialect-decls: Generate dialect class DECLARATION (like a .h)
#   Output: "class GaweeDialect : public mlir::Dialect { ... };"
# -I: Include search path, so tblgen can find "mlir/IR/OpBase.td" etc.
$TBLGEN --gen-dialect-decls \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.h.inc

# --gen-dialect-defs: Generate dialect class DEFINITION (like a .cpp)
#   Output: "void GaweeDialect::initialize() { addOperations<...>(); }"
$TBLGEN --gen-dialect-defs \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.cpp.inc

# --gen-op-decls: Generate op class DECLARATIONS (like a .h)
#   Output: "class ConvOp { ... }; class ReluOp { ... };" etc.
#   Includes accessor methods like getWeight(), getStrides(), etc.
$TBLGEN --gen-op-decls \
  -I $LLVM_DIR/include \
  -I include \
  -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.h.inc

# --gen-op-defs: Generate op class DEFINITIONS (like a .cpp)
#   Output: parser, printer, verifier, accessor implementations for each op.
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
# cmake does NOT compile -- it generates Makefiles (a build plan).
# -DMLIR_DIR: Tell cmake where MLIRConfig.cmake is, so find_package(MLIR) works.
# -DCMAKE_EXPORT_COMPILE_COMMANDS=ON: Generate compile_commands.json for IDE support.

rm -rf build
mkdir build
cd build

cmake .. -DMLIR_DIR="$LLVM_DIR/lib/cmake/mlir" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Symlink compile_commands.json to project root for IDE support
ln -sf "$SCRIPT_DIR/build/compile_commands.json" "$SCRIPT_DIR/compile_commands.json"

echo ""
echo "=== Step 3: Build ==="
# make actually compiles .cpp files + generated .inc files into binaries.

make

echo ""
echo "=== Done! ==="
