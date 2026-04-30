# Quiz: NLP 모델 AOT 파이프라인 확장

## Q1. ONNX Dynamic Shape Binding

ONNX 모델의 입력 shape에 `dim_param = "batch_size"`가 설정되어 있다.
이를 concrete value 1로 교체하는 Python 코드를 완성하시오.

```python
def bind_static_shapes(model, shape_overrides):
    param_map = {}
    for inp in model.graph.input:
        if inp.name not in shape_overrides:
            continue
        target = shape_overrides[inp.name]
        dims = inp.type.tensor_type.shape.dim
        for dim, val in zip(dims, target):
            if dim.HasField("dim_param"):
                param_map[________] = val     # (a) dim_param → value 매핑
            dim.ClearField("dim_param")
            dim.dim_value = val

    # output shape에도 적용
    for vi in list(model.graph.output):
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField("dim_param") and dim.dim_param in param_map:
                ________ = param_map[dim.dim_param]  # (b) 값 먼저 저장
                dim.ClearField("dim_param")
                dim.dim_value = ________              # (c) 저장한 값 사용
```

**빈칸**: (a), (b), (c)를 채우시오.

---

## Q2. MLIR Pass 순서

다음 pass들의 올바른 순서를 고르시오:

A. `MathToLibm`
B. `LLVMRequestCWrappers`
C. `FuncToLLVM`
D. `MathToLLVM`

**왜 B가 A보다 먼저 와야 하는가?**

> 답:
> 순서: ________ → ________ → ________ → ________
> 이유: ________________________________________________________________

---

## Q3. AggregatedOpInterface Decomposition

`linalg.softmax`를 분해하는 MLIR pass를 작성하라.
빈칸을 채우시오.

```cpp
struct DecomposePattern
    : public OpInterfaceRewritePattern<________> {  // (a) 어떤 인터페이스?
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(________ op,           // (b) 같은 인터페이스
                                PatternRewriter &rewriter) const override {
    FailureOr<SmallVector<Value>> result = op._________(rewriter);  // (c) 분해 메서드
    if (failed(result))
      return failure();
    rewriter.________(op, *result);  // (d) 원래 op를 결과로 교체
    return success();
  }
};
```

**빈칸**: (a), (b), (c), (d)를 채우시오.

---

## Q4. Scalar MemRef Regex

`memref<f32, strided<[], offset: ?>>` 를 파싱해야 한다.
ranked memref regex: `memref<([0-9\?x]+)x([a-z0-9]+)(?:,[^>]*)?>` 가 매칭에 실패하는 이유를 설명하시오.

> 답: ______________________________________________

scalar memref를 매칭하는 regex를 작성하시오:

> 답: `memref<________>`

---

## Q5. Initializer 분류

ONNX initializer `position_ids: tensor<0xi64>` (shape=[0], dtype=INT64)가 있다.

1. MLIR emitter에서 이 initializer는 `arith.constant`가 되는가, 함수 인수가 되는가?
2. 그 이유는?

> 답:
> 1. ________
> 2. ________________________________________________________________

---

## 정답

<details>
<summary>클릭하여 정답 확인</summary>

### Q1
- (a) `dim.dim_param`
- (b) `val` (또는 별도 변수에 저장)
- (c) `val`
- 핵심: `ClearField` 후에는 `dim.dim_param`이 빈 문자열이 되므로 값을 먼저 저장해야 함

### Q2
- 순서: B → A → D → C
- 이유: `LLVMRequestCWrappers`가 기존 `func.func`(forward)에만 `llvm.emit_c_interface` 속성을 추가함. 이후 `MathToLibm`이 `func.func @erff` 등의 선언을 생성하는데, 이 선언에는 wrapper 속성이 붙지 않음. 만약 순서가 반대면 libm 함수에도 `__mlir_ciface_` wrapper가 생성되어 링크 에러 발생.

### Q3
- (a) `linalg::AggregatedOpInterface`
- (b) `linalg::AggregatedOpInterface`
- (c) `decomposeOperation`
- (d) `replaceOp`

### Q4
- 이유: ranked regex는 `([0-9\?x]+)x([a-z0-9]+)` 패턴을 요구 — shape 뒤에 `x`와 dtype이 와야 함. 하지만 `memref<f32, ...>`는 shape 없이 바로 dtype이 와서 매칭 불가.
- scalar regex: `memref<([a-z0-9]+)(?:,[^>]*)?>` (shape 부분 없음)

### Q5
1. 함수 인수가 됨
2. `tensor<0xi64>`는 element가 0개라서 data가 비어있음. emitter의 `collectI64TensorLiteral`은 값이 있는 integer tensor만 `i64TensorLiterals`에 추가하고 `arith.constant`로 emit. 빈 tensor는 건너뛰어서 함수 인수로 분류됨.

</details>
