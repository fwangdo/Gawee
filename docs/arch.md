## pass 구조  
```text
│   ├── passes
│   │   ├── canonicalize.py
│   │   ├── constant_folding.py
│   │   ├── conv_add_folding.py
│   │   ├── conv_bn_folding.py
│   │   ├── elim_identity.py
│   │   ├── errors.py
│   │   ├── folder.py
│   │   ├── fusion.py
│   │   ├── inplace_marking.py
│   │   └── passer.py
```

## 현재 구현된 최적화
- Conv + BatchNorm Folding
- Conv + Bias(Add) Folding
- Constant Folding (Add / Mul / Relu)
- Identity Elimination
- FX bookkeeping node 제거(getitem / getattr)