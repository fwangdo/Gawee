# gawee-opt Tool - Summary

## What You Need to Understand

### 1. What is an MLIR opt Tool?

An "opt" tool (optimizer) is a command-line program that:
- Reads MLIR files (.mlir)
- Runs transformation passes on them
- Outputs the transformed IR

```bash
gawee-opt [options] <input.mlir>
gawee-opt --convert-gawee-to-linalg input.mlir  # Run our pass
gawee-opt --help                                 # List all options
```

### 2. Tool Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    gawee-opt                            │
├─────────────────────────────────────────────────────────┤
│  1. Register Dialects   →  What ops can be parsed       │
│  2. Register Passes     →  What transformations exist   │
│  3. MlirOptMain()       →  MLIR's standard opt driver   │
└─────────────────────────────────────────────────────────┘
```

### 3. Key Components

#### Dialect Registration
```cpp
DialectRegistry registry;
registry.insert<gawee::GaweeDialect>();    // Our dialect
registry.insert<linalg::LinalgDialect>();  // Target dialect
registry.insert<tensor::TensorDialect>();  // For tensor.empty
registry.insert<arith::ArithDialect>();    // For arith.constant
registry.insert<func::FuncDialect>();      // For func.func
```

**Why needed?** MLIR only understands ops from registered dialects.

#### Pass Registration
```cpp
PassPipelineRegistration<>(
    "convert-gawee-to-linalg",           // CLI flag name
    "Lower Gawee dialect to Linalg",     // Description
    [](OpPassManager &pm) {
      pm.addPass(gawee::createGaweeToLinalgPass());
    });
```

**Why needed?** Makes the pass available via `--convert-gawee-to-linalg`.

#### Dependent Dialects in Pass
```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<linalg::LinalgDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
}
```

**Why needed?** Pass creates ops from these dialects - they must be loaded.

### 4. Build System Requirements

#### CMake Libraries
```cmake
target_link_libraries(gawee-opt
  GaweeDialect        # Our dialect
  GaweeConversion     # Our pass
  MLIROptLib          # MlirOptMain
  MLIRParser          # Parse .mlir files
  MLIRPass            # Pass infrastructure
  MLIRRewrite         # Pattern rewriting
  MLIRTransforms      # Standard transforms
  MLIRIR              # Core IR
  # ... dialect libraries
)
```

#### RTTI Setting
```cmake
# LLVM is built without RTTI - we must match
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
```

**Why?** RTTI mismatch causes linker errors (`typeinfo` undefined symbols).

### 5. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `dialect not registered` | Dialect not in registry | Add `registry.insert<Dialect>()` |
| `op isn't known in this MLIRContext` | Pass creates op from unloaded dialect | Add `getDependentDialects()` |
| `typeinfo undefined` | RTTI mismatch | Add `-fno-rtti` flag |
| `undefined symbol` | Missing library | Add library to `target_link_libraries` |

### 6. Usage Examples

```bash
# Just parse and print (verify syntax)
./build/gawee-opt test/simple_test.mlir

# Run conversion pass
./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir

# Save output to file
./build/gawee-opt --convert-gawee-to-linalg test/simple_test.mlir -o output.mlir

# Show available passes
./build/gawee-opt --list-passes
`

## Key Takeaways

1. **Dialects must be registered** before parsing or creating their ops
2. **Passes must declare dependent dialects** via `getDependentDialects()`
3. **RTTI must match** between your code and LLVM/MLIR libraries
4. **MlirOptMain** handles CLI, parsing, pass execution - you just register things
