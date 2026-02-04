# Linalg to Loops - Summary

## What You Need to Understand

### 1. The Full Pipeline

```
Gawee IR
    │
    ▼ (convert-gawee-to-linalg)
Linalg on Tensors        ← immutable, SSA values
    │
    ▼ (one-shot-bufferize)
Linalg on MemRefs        ← mutable, memory buffers
    │
    ▼ (convert-linalg-to-loops)
SCF Loops                ← explicit for loops
```

### 2. Why Bufferization?

**Tensors** = immutable, like values
```mlir
%result = linalg.add ins(%a, %b) outs(%empty) -> tensor<2x3xf32>
```

**MemRefs** = mutable, like memory pointers
```mlir
linalg.add ins(%a, %b) outs(%alloc)  // writes INTO %alloc
```

Loops need actual memory to load/store. Bufferization:
1. Allocates memory (`memref.alloc`)
2. Converts tensor operations to memref operations

### 3. SCF Dialect (Structured Control Flow)

SCF = explicit loop constructs:

```mlir
scf.for %i = %c0 to %c2 step %c1 {
  scf.for %j = %c0 to %c3 step %c1 {
    %val = memref.load %input[%i, %j]
    // ... compute ...
    memref.store %result, %output[%i, %j]
  }
}
```

### 4. Loop Nesting by Op Type

| Op | Loops | Reason |
|----|-------|--------|
| add (2D) | 2 | One loop per dimension |
| relu (2D) | 2 | Elementwise = same shape |
| conv2d | 7 | batch, out_ch, H, W, in_ch, kH, kW |

### 5. Key Passes (Built-in to MLIR)

| Pass | What it does |
|------|--------------|
| `--one-shot-bufferize` | tensor → memref |
| `--convert-linalg-to-loops` | linalg → scf.for |
| `--convert-linalg-to-affine-loops` | linalg → affine.for (for polyhedral optimization) |

### 6. Using the Pipeline

**Option 1: Script**
```bash
./scripts/full_pipeline.sh test/simple_test.mlir
```

**Option 2: Manual chaining**
```bash
./build/gawee-opt --convert-gawee-to-linalg input.mlir \
  | mlir-opt --one-shot-bufferize="bufferize-function-boundaries" \
  | mlir-opt --convert-linalg-to-loops
```

## Key Takeaways

1. **Tensor vs MemRef**: Tensors are values, memrefs are memory locations
2. **Bufferization is required** before lowering to loops
3. **MLIR provides passes** - don't reinvent the wheel
4. **Loop depth = tensor rank** for elementwise ops
5. **Conv2d has many loops** - 7 for NCHW format
