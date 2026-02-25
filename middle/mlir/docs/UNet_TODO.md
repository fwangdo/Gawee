# UNet Support — What Needs to Be Done

## Current Status

The full pipeline (Frontend → MLIR → Linalg → LLVM) works for **ResNet-18**.
UNet JSON has been generated successfully (116 nodes after optimization).
The frontend already handles Cat and Interpolate (mapper, attr_extractor, passes).
Only the **MLIR side** needs new op support.

---

## UNet JSON Generated

```
Node count: 116
Ops:
  Conv: 47, Relu: 43, Add: 16, interpolate: 5, cat: 4, MaxPool: 1
```

## Frontend Fixes Applied

Two bugs were fixed to generate UNet JSON:

1. **`parser.py`** — `getattr`/`getitem` nodes got tensor shapes from ShapeProp
   instead of `None`. Fixed by checking op_type before `_get_shape()`.

2. **`translator.py`** — `interpolate` has optional `size`/`scale_factor` attrs
   that can be `None`. Fixed by skipping `None` attrs instead of asserting.

---

## New Ops Needed (MLIR only)

| Op | Count in UNet | Frontend Support | MLIR Support |
|----|--------------|-----------------|--------------|
| Conv | 47 | Yes | Yes |
| Relu | 43 | Yes | Yes |
| Add | 16 | Yes | Yes |
| MaxPool | 1 | Yes | Yes |
| **cat** | 4 | Yes (mapper, attr_extractor) | **No — needs new op** |
| **interpolate** | 5 | Yes (mapper, attr_extractor) | **No — needs new op** |

**Key new ops needed**:
1. **`gawee.cat`** — Concatenate tensors along a dimension (used in skip connections)
2. **`gawee.interpolate`** — Nearest/bilinear upsample (used in decoder)

---

## Step 3: Add New Ops to Gawee Dialect (TableGen)

**File**: `include/Gawee/GaweeOps.td`

For each new op, define:
- Input/output types
- Attributes (e.g., `axis` for Cat, `scale_factor`/`mode` for Interpolate)

### 3a. `gawee.cat`
```
Inputs:  variadic tensor inputs
Attrs:   axis (I64Attr)
Output:  1 tensor
```

### 3b. `gawee.interpolate`
```
Inputs:  1 tensor
Attrs:   scale_factor (F64ArrayAttr or I64ArrayAttr), mode (StrAttr)
Output:  1 tensor
```

After editing `.td`, rebuild to generate `.inc` files.

---

## Step 4: Add Lowering Patterns (Gawee → Linalg)

**File**: `lib/Conversion/GaweeToLinalg.cpp`

### 4a. CatOpLowering
- `gawee.cat` → `tensor.insert_slice` or manual `linalg.generic` that copies inputs into output

### 4b. InterpolateOpLowering
- `gawee.interpolate` → `linalg.generic` with index computation for nearest/bilinear sampling
- This is the hardest new lowering pattern

---

## Step 5: Update MLIREmitter

**File**: `lib/Emit/MLIREmitter.cpp`

Add `emitCat()` and `emitInterpolate()` to handle new JSON node types.

---

## Step 6: Test Full Pipeline

```bash
# JSON → MLIR
./build/gawee-translate jsondata/graph.json

# MLIR → Linalg
./build/gawee-translate jsondata/graph.json | ./build/gawee-opt --convert-gawee-to-linalg

# Full pipeline
./build/gawee-translate jsondata/graph.json | ./build/gawee-opt --gawee-to-llvm
```

---

## Recommended Order

```
Step 0: Fix frontend parser bug                    ← Python, small fix
Step 1: Generate UNet JSON & inspect ops           ← Python, verify
Step 2: Confirm which ops are actually missing      ← Analysis
Step 3: Add gawee.cat to dialect                   ← TableGen + rebuild
Step 4a: Write CatOpLowering                       ← C++, moderate
Step 5a: Add emitCat to MLIREmitter                ← C++, easy
Step 6a: Test Cat end-to-end                       ← verify
Step 3b: Add gawee.interpolate to dialect          ← TableGen + rebuild
Step 4b: Write InterpolateOpLowering               ← C++, hard
Step 5b: Add emitInterpolate to MLIREmitter        ← C++, easy
Step 6b: Test full UNet pipeline                   ← verify
```

---

## Difficulty Assessment

| Task | Difficulty | Why |
|------|-----------|-----|
| Fix parser bug (Step 0) | Easy | Small conditional check |
| Cat op definition (Step 3a) | Easy | Similar to Add but variadic |
| Cat lowering (Step 4a) | Medium | Need to handle `tensor.insert_slice` or loop over inputs |
| Interpolate op definition (Step 3b) | Easy | Straightforward TableGen |
| Interpolate lowering (Step 4b) | Hard | Index math for bilinear sampling in `linalg.generic` |
| Emitter updates (Step 5) | Easy | Follow existing patterns |
