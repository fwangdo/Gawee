# Phase 8: Extension Checklist

## Prerequisites
- [ ] Completed studying Phases 2-7
- [ ] Understand the patterns in existing code

## Files to Modify

### 1. Dialect Definition (TableGen)
**File:** `include/Gawee/GaweeOps.td`
**Guide:** `docs/extension/01_GaweeOps_Extension.td`

- [ ] Add `Gawee_MaxPoolOp` definition
- [ ] Add `Gawee_BatchNormOp` definition
- [ ] (Optional) Add bias to `Gawee_ConvOp`
- [ ] Run `./build.sh` to regenerate
- [ ] Verify generated files in `include/Gawee/generated/`

### 2. Lowering Pass
**File:** `lib/Conversion/GaweeToLinalg.cpp`
**Guide:** `docs/extension/02_GaweeToLinalg_Extension.cpp`

- [ ] Add `MaxPoolOpLowering` pattern
- [ ] Add `BatchNormOpLowering` pattern
- [ ] Register patterns in `runOnOperation()`
- [ ] Update `getDependentDialects()` if needed (e.g., math dialect)
- [ ] Test: `./build/gawee-opt --convert-gawee-to-linalg test/extension_test.mlir`

### 3. JSON Emitter
**File:** `lib/Emit/MLIREmitter.cpp`
**Header:** `include/Emit/MLIREmitter.h`
**Guide:** `docs/extension/03_MLIREmitter_Extension.cpp`

- [ ] Add `emitMaxPool()` method
- [ ] Add `emitBatchNorm()` method
- [ ] Update first pass to collect BN parameters (if needed)
- [ ] Update `emitNode()` dispatch
- [ ] Add declarations to header file
- [ ] Test: `./build/gawee-translate test/extension_graph.json`

### 4. Build System (if adding new dialects)
**File:** `CMakeLists.txt`

- [ ] Add `MLIRMathDialect` (for BatchNorm sqrt)
- [ ] Add `MLIRMathToLLVM` (for LLVM lowering)

**File:** `tools/gawee-opt.cpp`

- [ ] Register Math dialect
- [ ] Register Math bufferization interfaces (if needed)
- [ ] Add Math to LLVM conversion in pipeline

## Testing Progression

```bash
# Step 1: Test dialect parses correctly
./build/gawee-opt test/extension_test.mlir

# Step 2: Test lowering to Linalg
./build/gawee-opt --convert-gawee-to-linalg test/extension_test.mlir

# Step 3: Test full pipeline to LLVM
./build/gawee-opt --gawee-to-llvm test/extension_test.mlir

# Step 4: Test JSON translation
./build/gawee-translate test/extension_graph.json

# Step 5: Test JSON through full pipeline
./build/gawee-translate test/extension_graph.json | ./build/gawee-opt --gawee-to-llvm
```

## Linalg Ops Reference

| Gawee Op | Linalg Target | Notes |
|----------|---------------|-------|
| maxpool | `linalg.pooling_nchw_max` | Init with -inf |
| batchnorm | `linalg.generic` | Need math.sqrt |
| conv+bias | conv + `linalg.generic`/`linalg.add` | Broadcast bias |

## Common Pitfalls

1. **Forgot to register pattern** - Add to `patterns.add<>()` in pass
2. **Wrong output shape** - Calculate pooling output: `(H - K + 2P) / S + 1`
3. **BatchNorm broadcasting** - gamma/beta/mean/var are [C], input is [N,C,H,W]
4. **Missing dialect** - Register Math dialect for sqrt
5. **Bufferization fails** - Register bufferization interfaces for new dialects

## Success Criteria

- [ ] `gawee.maxpool` parses, lowers, and runs through full pipeline
- [ ] `gawee.batchnorm` parses, lowers, and runs through full pipeline
- [ ] Can translate a ResNet-like JSON with all ops
- [ ] Full pipeline: JSON → LLVM dialect works
