# Gawee MLIR Dialect

Educational MLIR dialect for neural network IR.

---

## Part 0: Before You Start

### System Requirements

| Item | Minimum | Recommended |
|------|---------|-------------|
| Disk Space | 30 GB | 50 GB |
| RAM | 8 GB | 16 GB |
| Build Time | 1-2 hours | - |

### Required Tools

Check if you have these installed:

```bash
# Check CMake (need 3.20+)
cmake --version
# If missing: brew install cmake

# Check Ninja (faster than make)
ninja --version
# If missing: brew install ninja

# Check Git
git --version
# If missing: brew install git

# Check C++ compiler
clang++ --version
# macOS: comes with Xcode Command Line Tools
# If missing: xcode-select --install
```

---

## Part 1: Build LLVM/MLIR from Source

### Step 1.1: Create workspace

```bash
# Create a dedicated directory (NOT inside Gawee project)
mkdir -p ~/mlir-workspace
cd ~/mlir-workspace
```

### Step 1.2: Clone LLVM repository

```bash
# This downloads ~2GB, takes 5-15 minutes depending on network
git clone --depth 1 https://github.com/llvm/llvm-project.git

# --depth 1 means shallow clone (only latest commit, saves disk space)
```

**Expected output:**
```
Cloning into 'llvm-project'...
remote: Enumerating objects: ...
Receiving objects: 100% ...
```

**Verify:**
```bash
ls llvm-project/mlir
# Should show: CMakeLists.txt, include, lib, test, tools, ...
```

### Step 1.3: Create build directory

```bash
cd llvm-project
mkdir build
cd build
```

### Step 1.4: Configure with CMake

```bash
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
```

**What each flag means:**
| Flag | Purpose |
|------|---------|
| `-G Ninja` | Use Ninja build system (faster than make) |
| `-DLLVM_ENABLE_PROJECTS=mlir` | Build MLIR (we don't need clang, lld, etc.) |
| `-DLLVM_TARGETS_TO_BUILD="host"` | Only build for your CPU (saves time) |
| `-DCMAKE_BUILD_TYPE=Release` | Optimized build (faster execution) |
| `-DLLVM_ENABLE_ASSERTIONS=ON` | Keep assertions (helpful for debugging) |

**Expected output (last few lines):**
```
-- Configuring done
-- Generating done
-- Build files have been written to: /Users/xxx/mlir-workspace/llvm-project/build
```

**If you see errors:**
| Error | Solution |
|-------|----------|
| `Could not find ninja` | `brew install ninja` |
| `CMake version too old` | `brew upgrade cmake` |
| `No C compiler found` | `xcode-select --install` |

### Step 1.5: Build MLIR

```bash
# Build only MLIR tools (not all of LLVM)
ninja mlir-opt mlir-tblgen mlir-cpu-runner

# This takes 30-90 minutes. Go get coffee.
# You'll see progress like: [1234/5678] Building CXX object...
```

**To use all CPU cores (faster but uses more memory):**
```bash
ninja -j$(sysctl -n hw.ncpu) mlir-opt mlir-tblgen mlir-cpu-runner
```

**If build fails with "out of memory":**
```bash
# Use fewer parallel jobs
ninja -j4 mlir-opt mlir-tblgen mlir-cpu-runner
```

### Step 1.6: Verify the build

```bash
# Check mlir-opt exists and runs
./bin/mlir-opt --version

# Expected output:
# LLVM (http://llvm.org/):
#   LLVM version 19.x.x (or similar)
#   Optimized build with assertions.
```

```bash
# Check mlir-tblgen exists
./bin/mlir-tblgen --help | head -5

# Expected output:
# USAGE: mlir-tblgen [options] <input file>
# ...
```

**Congratulations!** MLIR is built. Now let's set up environment variables.

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
export LLVM_BUILD_DIR="$HOME/mlir-workspace/llvm-project/build"
export PATH="$LLVM_BUILD_DIR/bin:$PATH"
export MLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir"
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
# Expected: /Users/xxx/mlir-workspace/llvm-project/build/bin/mlir-opt

which mlir-tblgen
# Expected: /Users/xxx/mlir-workspace/llvm-project/build/bin/mlir-tblgen

echo $MLIR_DIR
# Expected: /Users/xxx/mlir-workspace/llvm-project/build/lib/cmake/mlir
```

---

## Part 3: Generate Code from TableGen

Now we work in the Gawee project.

### Step 3.1: Navigate to mlir directory

```bash
cd /path/to/Gawee/middle/mlir
```

### Step 3.2: Create output directory for generated files

```bash
mkdir -p include/Gawee/generated
```

### Step 3.3: Run mlir-tblgen

```bash
# 3.3.1: Generate dialect declaration (.h.inc)
mlir-tblgen --gen-dialect-decls \
  -I $LLVM_BUILD_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.h.inc

# 3.3.2: Generate dialect definition (.cpp.inc)
mlir-tblgen --gen-dialect-defs \
  -I $LLVM_BUILD_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.cpp.inc

# 3.3.3: Generate op declarations
mlir-tblgen --gen-op-decls \
  -I $LLVM_BUILD_DIR/include \
  -I include \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.h.inc

# 3.3.4: Generate op definitions
mlir-tblgen --gen-op-defs \
  -I $LLVM_BUILD_DIR/include \
  -I include \
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
~/mlir-workspace/
└── llvm-project/
    └── build/
        └── bin/
            ├── mlir-opt         # MLIR optimizer tool
            ├── mlir-tblgen      # TableGen for MLIR
            └── mlir-cpu-runner  # JIT execution

Gawee/middle/mlir/
├── include/Gawee/
│   ├── GaweeDialect.td
│   ├── GaweeOps.td
│   ├── GaweeDialect.h
│   └── generated/           # Auto-generated
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
