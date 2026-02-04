# GaweeToLinalg Conversion Pass - Summary

## What You Need to Understand

### 1. Conversion Pass Architecture

A conversion pass transforms operations from one dialect to another (called "lowering").

**Three key components:**
- **ConversionPattern**: Describes how to rewrite ONE specific op type
- **ConversionTarget**: Defines which ops are "legal" after conversion
- **RewritePatternSet**: Collection of patterns to apply

```
gawee.conv  ──[ConvOpLowering]──>  linalg.conv_2d_nchw_fchw
gawee.relu  ──[ReluOpLowering]──>  linalg.generic
gawee.add   ──[AddOpLowering]───>  linalg.add
```

### 2. OpConversionPattern Structure

```cpp
struct MyOpLowering : public OpConversionPattern<gawee::MyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gawee::MyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // 1. Get location (for error messages)
    Location loc = op.getLoc();

    // 2. Get operands from adaptor (already-converted values)
    Value input = adaptor.getInput();

    // 3. Get attributes from op
    auto attr = op.getMyAttrAttr();  // Returns Attribute
    auto values = op.getMyAttr();    // Returns ArrayRef<int64_t>

    // 4. Create new ops
    Value result = rewriter.create<SomeOp>(loc, ...);

    // 5. Replace original op
    rewriter.replaceOp(op, result);

    return success();
  }
};
```

### 3. Key Concepts

| Concept | Meaning |
|---------|---------|
| `adaptor` | Provides already-converted operand values |
| `op` | The original operation being converted |
| `rewriter` | Used to create new ops and replace old ones |
| `Location` | Source location for debugging/errors |
| `LogicalResult` | Return `success()` or `failure()` |

### 4. TableGen → C++ Mapping

When you define in `.td`:
```tablegen
let arguments = (ins
  AnyTensor:$input,
  DenseI64ArrayAttr:$strides
);
```

TableGen generates:
- `adaptor.getInput()` → `Value` (the tensor)
- `op.getStrides()` → `ArrayRef<int64_t>` (the values)
- `op.getStridesAttr()` → `DenseI64ArrayAttr` (the attribute itself)

### 5. Linalg Patterns

**Destination-Passing Style**: Linalg ops write INTO an output tensor.

```cpp
// Step 1: Create empty output tensor
Value output = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);

// Step 2: Create linalg op with ins/outs
auto result = rewriter.create<linalg::SomeOp>(
    loc, resultType,
    /*ins=*/ValueRange{input1, input2},
    /*outs=*/ValueRange{output}
);
```

**linalg.generic**: Swiss army knife for custom elementwise ops.

```cpp
// For elementwise: identity maps, parallel iterators
SmallVector<AffineMap, 2> maps(2, AffineMap::getMultiDimIdentityMap(rank, ctx));
SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

rewriter.create<linalg::GenericOp>(
    loc, resultTypes, inputs, outputs, maps, iterTypes,
    [&](OpBuilder &b, Location loc, ValueRange args) {
      // args[0] = input element, args[1] = output element
      Value result = /* compute */;
      b.create<linalg::YieldOp>(loc, result);
    }
);
```

### 6. Pass Setup

```cpp
void runOnOperation() override {
  MLIRContext *ctx = &getContext();

  // 1. Define legal/illegal ops
  ConversionTarget target(*ctx);
  target.addLegalDialect<linalg::LinalgDialect>();
  target.addIllegalDialect<gawee::GaweeDialect>();

  // 2. Collect patterns
  RewritePatternSet patterns(ctx);
  patterns.add<ConvOpLowering, ReluOpLowering, AddOpLowering>(ctx);

  // 3. Run conversion
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

## Key Takeaways

1. **Adaptor vs Op**: Use `adaptor` for operands (values), `op` for attributes
2. **getXxx() vs getXxxAttr()**: One returns values, other returns the Attribute object
3. **Destination-passing**: Always create empty output tensor first for Linalg ops
4. **linalg.generic**: Use for custom elementwise ops with a body builder lambda
5. **Pattern registration**: Add all patterns to RewritePatternSet, then run conversion
