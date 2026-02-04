# Gawee MLIR Dialect

Educational MLIR dialect for neural network IR.

---

## Part 0: Current Status

### Already Completed

- [x] LLVM/MLIR installed at `~/llvm-install/`
- [x] TableGen files generated at `include/Gawee/generated/`
- [ ] Environment variables (see Part 2)
- [ ] Build Gawee MLIR library (see Part 4)

### Required Tools

```bash
# Verify these work:
cmake --version        # Need 3.20+
ninja --version        # Optional but faster
clang++ --version      # C++ compiler
```

---

## Part 1: LLVM/MLIR Installation

**Already installed at:** `~/llvm-install/`

### Verify installation

```bash
# Check mlir-opt works
~/llvm-install/bin/mlir-opt --version
# Expected: LLVM version 22.0.0git (or similar)

# Check mlir-tblgen works
~/llvm-install/bin/mlir-tblgen --help | head -3
```

---

## Part 2: Set Up Environment

### Step 2.1: Add to shell profile

Open your shell config file:
```bash
# For zsh (default on modern macOS)
nano ~/.zshrc

# For bash
nano ~/.bashrc
```

Add these lines at the end:
```bash
# MLIR/LLVM environment
export LLVM_INSTALL_DIR="$HOME/llvm-install"
export PATH="$LLVM_INSTALL_DIR/bin:$PATH"
export MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter` in nano).

### Step 2.2: Apply changes

```bash
source ~/.zshrc  # or ~/.bashrc
```

### Step 2.3: Verify environment

```bash
# These should work from any directory now
which mlir-opt
# Expected: /Users/hdy/llvm-install/bin/mlir-opt

which mlir-tblgen
# Expected: /Users/hdy/llvm-install/bin/mlir-tblgen

echo $MLIR_DIR
# Expected: /Users/hdy/llvm-install/lib/cmake/mlir
```

---

## Part 3: Generate Code from TableGen

**Status: Already completed.** Files exist at `include/Gawee/generated/`.

If you need to regenerate (after modifying `.td` files):

### Step 3.1: Navigate to mlir directory

```bash
cd /path/to/Gawee/middle/mlir
```

### Step 3.2: Create output directory (if needed)

```bash
mkdir -p include/Gawee/generated
```

### Step 3.3: Run mlir-tblgen

```bash
# 3.3.1: Generate dialect declaration (.h.inc)
mlir-tblgen --gen-dialect-decls \
  -I $LLVM_INSTALL_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.h.inc

# 3.3.2: Generate dialect definition (.cpp.inc)
mlir-tblgen --gen-dialect-defs \
  -I $LLVM_INSTALL_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.cpp.inc

# 3.3.3: Generate op declarations
mlir-tblgen --gen-op-decls \
  -I $LLVM_INSTALL_DIR/include \
  -I include \
  -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.h.inc

# 3.3.4: Generate op definitions
mlir-tblgen --gen-op-defs \
  -I $LLVM_INSTALL_DIR/include \
  -I include \
  -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.cpp.inc
```

**Verify generated files:**
```bash
ls -la include/Gawee/generated/
# Should show:
# GaweeDialect.h.inc
# GaweeDialect.cpp.inc
# GaweeOps.h.inc
# GaweeOps.cpp.inc
```

**If you see errors:**
| Error | Solution |
|-------|----------|
| `could not find include 'mlir/IR/OpBase.td'` | Check `$LLVM_BUILD_DIR` is set correctly |
| `unknown directive` | Your LLVM version may be too old, rebuild |

### Step 3.4: Examine generated code (educational)

```bash
# Look at generated op class
head -100 include/Gawee/generated/GaweeOps.h.inc

# You'll see C++ classes like:
# class ConvOp : public Op<ConvOp, ...> { ... }
```

---

## Part 4: Build Gawee MLIR Library

### Step 4.1: Create CMakeLists.txt

Create file `middle/mlir/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR LANGUAGES CXX)

# C++17 required for MLIR
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#===----------------------------------------------------------------------===#
# Find MLIR
#===----------------------------------------------------------------------===#

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# MLIR/LLVM CMake utilities
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Include paths
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee/generated)

#===----------------------------------------------------------------------===#
# Gawee Dialect Library
#===----------------------------------------------------------------------===#

add_mlir_dialect_library(GaweeDialect
  lib/Gawee/GaweeDialect.cpp

  DEPENDS
  MLIRGaweeIncGen  # Generated files dependency
)

# Link MLIR libraries
target_link_libraries(GaweeDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# Conversion Library
#===----------------------------------------------------------------------===#

add_mlir_library(GaweeConversion
  lib/Conversion/GaweeToLinalg.cpp

  DEPENDS
  GaweeDialect
)

target_link_libraries(GaweeConversion
  PUBLIC
  GaweeDialect
  MLIRLinalg
  MLIRTensor
  MLIRArith
  MLIRTransforms
)

#===----------------------------------------------------------------------===#
# Test Executable (optional)
#===----------------------------------------------------------------------===#

# add_executable(gawee-opt tools/gawee-opt.cpp)
# target_link_libraries(gawee-opt GaweeDialect GaweeConversion MLIROptLib)
```

### Step 4.2: Update include paths in source files

Edit `include/Gawee/GaweeDialect.h`:
```cpp
// Change:
#include "Gawee/GaweeDialect.h.inc"
// To:
#include "generated/GaweeDialect.h.inc"

// Change:
#include "Gawee/GaweeOps.h.inc"
// To:
#include "generated/GaweeOps.h.inc"
```

Edit `lib/Gawee/GaweeDialect.cpp`:
```cpp
// Change:
#include "Gawee/GaweeDialect.cpp.inc"
// To:
#include "generated/GaweeDialect.cpp.inc"

// Change:
#include "Gawee/GaweeOps.cpp.inc"
// To:
#include "generated/GaweeOps.cpp.inc"
```

### Step 4.3: Build

```bash
cd /path/to/Gawee/middle/mlir
mkdir build && cd build

cmake .. -DMLIR_DIR=$MLIR_DIR
make

# Or with Ninja (faster):
cmake .. -G Ninja -DMLIR_DIR=$MLIR_DIR
ninja
```

---

## Part 5: Verify Everything Works

### Step 5.1: Create a test MLIR file

Create `test/simple.mlir`:
```mlir
module {
  func.func @test_relu(%arg0: tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32> {
    %0 = "gawee.relu"(%arg0) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
    return %0 : tensor<1x64x112x112xf32>
  }
}
```

### Step 5.2: Parse with mlir-opt

```bash
# This should parse without errors once your dialect is registered
mlir-opt test/simple.mlir --load-dialect=gawee
```

---

## Troubleshooting

### "mlir-tblgen: command not found"
```bash
# Check if PATH is set
echo $PATH | grep llvm

# If not, re-run:
source ~/.zshrc
```

### "Cannot find MLIR" during CMake
```bash
# Verify MLIR_DIR is correct
ls $MLIR_DIR/MLIRConfig.cmake

# If file doesn't exist, LLVM build may have failed
# Go back to Part 1, Step 1.5 and rebuild
```

### Build fails with "undefined reference"
```bash
# Usually means missing library linkage
# Check CMakeLists.txt target_link_libraries
```

### Out of disk space
```bash
# Clean up LLVM build artifacts (keeps binaries)
cd ~/mlir-workspace/llvm-project/build
ninja clean

# Or remove object files manually
find . -name "*.o" -delete
```

---

## File Structure After Setup

```
~/llvm-install/              # Your LLVM/MLIR installation
├── bin/
│   ├── mlir-opt             # MLIR optimizer tool
│   ├── mlir-tblgen          # TableGen for MLIR
│   └── mlir-cpu-runner      # JIT execution
├── include/                 # MLIR headers
└── lib/cmake/mlir/          # CMake config

Gawee/middle/mlir/
├── include/Gawee/
│   ├── GaweeDialect.td
│   ├── GaweeOps.td
│   ├── GaweeDialect.h
│   └── generated/           # Auto-generated (already done)
│       ├── GaweeDialect.h.inc
│       ├── GaweeDialect.cpp.inc
│       ├── GaweeOps.h.inc
│       └── GaweeOps.cpp.inc
├── lib/
│   ├── Gawee/
│   │   └── GaweeDialect.cpp
│   └── Conversion/
│       └── GaweeToLinalg.cpp
├── CMakeLists.txt
├── build/                   # Build output
└── README.md
```

---

## Next Steps (Learning Path)

| Order | Task | Goal |
|-------|------|------|
| 1 | Complete setup above | Environment works |
| 2 | Read generated `.inc` files | Understand what TableGen produces |
| 3 | Add `Gawee_FlattenOp` to `GaweeOps.td` | Learn to define ops |
| 4 | Implement `ReluOpLowering` | Learn pattern rewriting |
| 5 | Write JSON → MLIR importer | Connect your Python frontend |
| 6 | Run full pipeline | Gawee → Linalg → LLVM → execute |

---

## Resources

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/) - **Highly recommended**
- [Defining Dialects](https://mlir.llvm.org/docs/DefiningDialects/)
- [TableGen Programmer's Reference](https://llvm.org/docs/TableGen/ProgRef.html)
