# Gawee MLIR Dialect

Educational MLIR dialect for neural network IR.

## File Structure

```
mlir/
├── include/Gawee/
│   ├── GaweeDialect.td   # Dialect definition (TableGen)
│   ├── GaweeOps.td       # Operations definition (TableGen)
│   └── GaweeDialect.h    # C++ header
├── lib/
│   ├── Gawee/
│   │   └── GaweeDialect.cpp    # Dialect implementation
│   └── Conversion/
│       └── GaweeToLinalg.cpp   # Lowering pass
└── README.md
```

## Prerequisites

### 1. Build LLVM/MLIR

MLIR is part of LLVM. You need to build it from source:

```bash
# Clone LLVM (this is large, ~2GB)
git clone https://github.com/llvm/llvm-project.git
cd llvm-project

# Create build directory
mkdir build && cd build

# Configure (enable MLIR)
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON

# Build (this takes a while, 30min - 2hrs depending on machine)
ninja

# Optionally install
sudo ninja install
```

### 2. Set up environment

```bash
export LLVM_BUILD_DIR=/path/to/llvm-project/build
export PATH=$LLVM_BUILD_DIR/bin:$PATH
```

## Build Steps

### Step 1: Generate C++ from TableGen

```bash
# Generate dialect declaration
mlir-tblgen --gen-dialect-decls \
  include/Gawee/GaweeDialect.td \
  -I $LLVM_BUILD_DIR/include \
  -o include/Gawee/GaweeDialect.h.inc

# Generate dialect definition
mlir-tblgen --gen-dialect-defs \
  include/Gawee/GaweeDialect.td \
  -I $LLVM_BUILD_DIR/include \
  -o include/Gawee/GaweeDialect.cpp.inc

# Generate op declarations
mlir-tblgen --gen-op-decls \
  include/Gawee/GaweeOps.td \
  -I $LLVM_BUILD_DIR/include \
  -I include \
  -o include/Gawee/GaweeOps.h.inc

# Generate op definitions
mlir-tblgen --gen-op-defs \
  include/Gawee/GaweeOps.td \
  -I $LLVM_BUILD_DIR/include \
  -I include \
  -o include/Gawee/GaweeOps.cpp.inc
```

### Step 2: Build with CMake

Create a CMakeLists.txt (you'll need to write this - see hints below).

```bash
mkdir build && cd build
cmake .. -DMLIR_DIR=$LLVM_BUILD_DIR/lib/cmake/mlir
make
```

## CMakeLists.txt Hints

```cmake
cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

# Include MLIR's CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# TODO: Add mlir-tblgen rules for .td files
# TODO: Add library for GaweeDialect
# TODO: Add executable for testing
```

## Learning Path

### Level 1: Understand TableGen
1. Read `GaweeDialect.td` - understand dialect definition
2. Read `GaweeOps.td` - understand how ops are defined
3. Run `mlir-tblgen` manually, inspect generated `.inc` files

### Level 2: Implement Missing Ops
1. Add `Gawee_BatchNormOp` to `GaweeOps.td`
2. Add `Gawee_ConcatOp` for skip connections
3. Re-generate and verify compilation

### Level 3: Implement Lowering
1. Complete `ConvOpLowering` in `GaweeToLinalg.cpp`
2. Complete `ReluOpLowering`
3. Test with simple MLIR input

### Level 4: End-to-End
1. Write JSON -> MLIR importer (read your graph.json, emit MLIR)
2. Run full pipeline: Gawee -> Linalg -> LLVM
3. Execute with `mlir-cpu-runner`

## Testing Your Dialect

Create a test file `test.mlir`:

```mlir
// RUN: mlir-opt %s --convert-gawee-to-linalg | FileCheck %s

module {
  func.func @test_relu(%input: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    %0 = gawee.relu %input : tensor<1x64x112x112xf32>
    return %0 : tensor<1x64x112x112xf32>
  }
}

// CHECK: linalg.generic
```

Run:
```bash
mlir-opt test.mlir --convert-gawee-to-linalg
```

## Resources

- [MLIR Tutorial](https://mlir.llvm.org/docs/Tutorials/)
- [Toy Language Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [ONNX-MLIR](https://github.com/onnx/onnx-mlir) - Production example
- [Torch-MLIR](https://github.com/llvm/torch-mlir) - Another production example
