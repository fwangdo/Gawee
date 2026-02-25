# gawee

A deep learning compiler project that converts PyTorch models into a custom IR, performs graph analysis and optimization, and lowers through an MLIR-based middle-end pipeline (Linalg → SCF → LLVM IR). Designed to tackle real-world problems found in production DL compilers (e.g., TVM).

`gawee`는 PyTorch로 작성된 딥러닝 모델을 자체 IR로 변환한 뒤, 그래프 분석 및 최적화(graph optimization)를 수행하고,
MLIR 기반 미들엔드를 통해 Linalg → SCF → LLVM IR까지 lowering하는 딥러닝 컴파일러 프로젝트이며,
실무 딥러닝 컴파일러(예: TVM 계열)가 실제로 수행하는 문제를 직접 설계 및 구현하는 것을 목적으로 개발되었습니다.

---

## Target Models / 대상 모델

- **ResNet-18**
  - Based on the standard ImageNet ResNet-18 architecture.
  - Goal: reduce graph nodes by fusing Conv / BatchNorm.
  - ImageNet용 표준 ResNet-18 구조를 기반으로 함. Conv / BatchNorm을 fusion하여 그래프 노드를 줄이는 것을 목표로 함.
- **UNet**
  - Based on the standard ImageNet UNet architecture.
  - Goal: eliminate redundant Identity ops and removable Python ops (e.g., getitem, getattr).
  - ImageNet용 표준 Unet 구조를 기반으로 함. 중복되는 Identity 함수를 제거하고, 제거가능한 파이썬 연산(e.g., getitem, getattr)들을 최대한 제거하는 것을 목표로 함.
- [Usage & Evaluation / 실행 및 평가 방법](docs/usage.md)

---

## Pipeline / 파이프라인

```
PyTorch Model → FX Graph → Gawee IR → JSON → MLIR(Gawee Dialect) → Linalg → SCF → LLVM
```

### Frontend (Python) / 프론트엔드

- Parse torch fx graph into Gawee IR / torch fx 그래프를 Gawee IR로 파싱
- Measure baseline cost using predefined costs / 사전 정의된 cost를 기반으로 베이스라인의 cost 측정
- Optimize using passes defined in Gawee IR / Gawee IR에서 정의된 pass를 기반으로 최적화 수행
- Export IR as JSON / IR을 JSON 형태로 저장

### Middle-end (C++ / MLIR) / 미들엔드

- Convert JSON graph to MLIR Gawee Dialect (MLIREmitter) / JSON 그래프를 MLIR Gawee Dialect으로 변환
- Gawee Dialect → Linalg Dialect conversion (GaweeToLinalg) / Gawee Dialect → Linalg Dialect 변환
- Multi-stage lowering: Linalg → Bufferization → SCF loops → LLVM Dialect
- Two CLI tools: `gawee-opt`, `gawee-translate`

---

## Key Concepts / 프로젝트에서 다루는 핵심 개념

### 1. Gawee IR Design / Gawee IR 설계

A custom IR that explicitly represents only the information needed for graph analysis and optimization.

**그래프 분석과 최적화를 위해 필요한 정보만을 명시적으로 표현하는 자체 IR**를 정의.

- Clear separation of operation nodes and data flow / 연산 노드와 데이터 흐름의 명확한 분리
- Explicit shape / dtype / layout / data representation
- Graph: Nodes (ops) / Values (tensors)
- Node: op type / input / output / attributes / fx Node
- Value: shape / dtype / producer / consumers / data (only for constants)

---

### 2. Graph Analysis / 그래프 분석

Analysis performed for optimization:

- Shape inference
- Constant propagation
- Graph traversal (topological order)
- Cost estimation:
  - FLOPs
  - Memory access estimation (read/write)

---

### 3. Frontend Optimization / 프론트엔드 최적화

Only **graph-level optimizations** are performed in the frontend. / 프론트엔드에서는 **그래프 레벨 최적화**만 수행.

- Constant Folding — evaluate constant subgraphs at compile time / 상수 서브그래프를 컴파일 타임에 계산
- Operator Fusion — combine consecutive op patterns into fused operators / 연속된 연산 패턴을 하나의 fused operator로 결합
  - e.g., Conv + BatchNorm, Conv + Add
- Eliminate Python ops from fx / fx에 존재하는 파이썬 연산 제거
- [Optimization pass details / 구현된 최적화 패스 설명](docs/arch.md)

---

### 4. MLIR Gawee Dialect / MLIR Gawee Dialect 정의

Custom MLIR Dialect for DL ops defined using TableGen. / **TableGen을 사용하여 딥러닝 연산에 대응하는 커스텀 MLIR Dialect**를 정의.

- Ops under `gawee` namespace: Conv2D / ReLU / Add / BatchNorm / MaxPool / AdAvgPool / Flatten / Linear
- Input/output types and attributes (stride, padding, dilation, etc.) declared in TableGen
- C++ boilerplate auto-generated from TableGen

---

### 5. JSON → MLIR Conversion (MLIREmitter) / JSON → MLIR 변환

Converts the frontend JSON graph to MLIR Gawee Dialect ops. / 프론트엔드에서 출력한 **JSON 그래프를 MLIR Gawee Dialect 연산으로 변환**.

- Parse inputs, outputs, weights, and nodes from JSON
- Register weight tensors as function arguments
- Traverse nodes in topological order to emit `gawee.*` ops
- Output an MLIR module as `func.func @main(...)`

---

### 6. Gawee → Linalg Lowering

Convert Gawee Dialect ops to Linalg Dialect using OpConversionPattern. / **OpConversionPattern을 사용하여 Gawee Dialect 연산을 Linalg Dialect으로 변환**.

- `gawee.conv` → `linalg.conv_2d_nchw_fchw` (with padding)
- `gawee.relu` → `linalg.generic` (max(x, 0) body)
- `gawee.add` → `linalg.add`
- Lowering patterns for BatchNorm / MaxPool / AdAvgPool / Flatten / Linear
- Uses ConversionTarget, TypeConverter dialect conversion framework

---

### 7. Multi-stage Lowering

Multi-stage lowering pipeline from Linalg to LLVM IR. / Linalg에서 **LLVM IR까지 다단계 lowering 파이프라인**을 구성.

- `--convert-gawee-to-linalg`: Gawee → Linalg
- `--gawee-to-loops`: Gawee → Linalg → Bufferization → SCF loops
- `--gawee-to-llvm`: Gawee → Linalg → Bufferize → SCF → LLVM (full pipeline)
- Includes tensor → memref conversion (bufferization)

---

## Optimization Results / 최적화 결과

### ResNet-18
```
Before: 69 nodes → After: 49 nodes
  - ConvBNFolding: 20 applications
  - Memory reads:  37.2MB → 27.3MB (26.7% reduction)
  - Memory writes: 32.9MB → 23.0MB (30.2% reduction)
```

### UNet
```
Before: 196 nodes → After: 116 nodes
  - IdentityElimination: 12, ConvBNFolding: 46, PythonOpElimination: 22
  - Memory reads:  136.6MB → 94.9MB (30.5% reduction)
  - Memory writes: 116.0MB → 83.7MB (27.9% reduction)
```

---

## Project Structure / 프로젝트 구조

```
gawee/
├── gawee_ir/                  # Frontend (Python) / 프론트엔드
│   ├── graph.py               #   Gawee IR definition (Graph / Node / Value)
│   ├── parser.py              #   PyTorch FX → Gawee IR conversion
│   ├── mapper.py              #   PyTorch op → Gawee op mapping
│   ├── translator.py          #   Gawee IR → JSON conversion
│   ├── analysis/              #   Shape inference, Cost analysis
│   └── passes/                #   Optimization passes (Conv-BN folding, etc.)
├── middle/mlir/               # Middle-end (C++ / MLIR) / 미들엔드
│   ├── include/Gawee/         #   TableGen definitions (Dialect, Ops)
│   ├── lib/Gawee/             #   Dialect registration
│   ├── lib/Conversion/        #   Gawee → Linalg conversion patterns
│   ├── lib/Emit/              #   JSON → MLIR conversion (MLIREmitter)
│   └── tools/                 #   gawee-opt, gawee-translate
├── scripts/                   # Scripts
├── jsondata/                  # Frontend output JSON
└── docs/                      # Documentation
```

---

## References / 참고

- PyTorch fx documentation
- ONNX specification
- TVM architecture documentation
- MLIR documentation (Dialects, TableGen, Conversion)
