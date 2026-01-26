## 현재 구현된 최적화
- Conv + BatchNorm Folding
- Conv + Bias(Add) Folding
- Constant Folding (Add / Mul / Relu)
- Identity Elimination
- FX bookkeeping node 제거(getitem / getattr)

## 향후 확장 예정
- Conv + ReLU fusion
- Transpose / Reshape canonicalization
- MLIR 기반 미들엔드