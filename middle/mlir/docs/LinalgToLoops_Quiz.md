# Linalg to Loops - Quiz

## Conceptual Questions

### Q1: Pipeline Order
What is the correct order of the lowering pipeline?

A) Gawee → Loops → Linalg → Bufferize
B) Gawee → Bufferize → Linalg → Loops
C) Gawee → Linalg → Bufferize → Loops
D) Gawee → Linalg → Loops → Bufferize

### Q2: Why Bufferize?
Why can't we convert Linalg directly to loops without bufferization?

A) MLIR doesn't support it
B) Loops need memory (memref) to load/store, not immutable tensors
C) It's just a convention
D) Tensors are slower

### Q3: Tensor vs MemRef
What's the key difference between tensor and memref?

A) Tensor is for GPU, memref is for CPU
B) Tensor is immutable (value), memref is mutable (memory)
C) Tensor is 2D, memref can be any dimension
D) No difference, just different names

### Q4: Loop Count
How many nested loops does a 2D elementwise operation (like add on tensor<2x3xf32>) generate?

A) 1
B) 2
C) 3
D) 6

### Q5: Conv2D Loops
Why does conv2d generate 7 nested loops for NCHW format?

A) It's a bug
B) One loop for: batch, output_channels, output_H, output_W, input_channels, kernel_H, kernel_W
C) It's inefficient code
D) MLIR always generates 7 loops

### Q6: Built-in Passes
Which of these passes is NOT built into MLIR?

A) --convert-linalg-to-loops
B) --one-shot-bufferize
C) --convert-gawee-to-linalg
D) --convert-linalg-to-affine-loops

### Q7: Bufferization Output
What does `memref.alloc()` do in the bufferized output?

A) Declares a variable
B) Allocates memory for the output tensor
C) Creates a new tensor
D) Initializes values to zero

---

## Code Reading

### Q8: What does this SCF code do?

```mlir
scf.for %i = %c0 to %c2 step %c1 {
  scf.for %j = %c0 to %c3 step %c1 {
    %a = memref.load %input[%i, %j]
    %b = memref.load %other[%i, %j]
    %sum = arith.addf %a, %b : f32
    memref.store %sum, %output[%i, %j]
  }
}
```

A) Matrix multiplication
B) Elementwise addition of 2x3 tensors
C) Convolution
D) Reduction sum

---

## Answers

```
Q1: C - Gawee → Linalg → Bufferize → Loops
Q2: B - Loops need memory to load/store
Q3: B - Tensor is immutable, memref is mutable
Q4: B - 2 loops (one per dimension)
Q5: B - One loop per: batch, out_ch, H, W, in_ch, kH, kW
Q6: C - convert-gawee-to-linalg is OUR pass, not built-in
Q7: B - Allocates memory for output
Q8: B - Elementwise add of 2x3 tensors (2 outer loop = 2, 3 inner loop = 3)
```
