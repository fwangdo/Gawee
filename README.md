# gawee

`gawee`는 PyTorch로 작성된 딥러닝 모델을 자체 IR로 변환한 뒤, 그래프 분석 및 최적화(graph optimization)를 수행하고,
MLIR 기반 미들엔드를 통해 Linalg → SCF → LLVM IR까지 lowering하는 딥러닝 컴파일러 프로젝트이며,
실무 딥러닝 컴파일러(예: TVM 계열)가 실제로 수행하는 문제를 직접 설계 및 구현하는 것을 목적으로 개발되었습니다.

---

## 대상 모델

- **ResNet-18**
  - ImageNet용 표준 ResNet-18 구조를 기반으로 함.
  - Conv / BatchNorm을 fusion하여 그래프 노드를 줄이는 것을 목표로 함.
- **Unet**
  - ImageNet용 표준 Unet 구조를 기반으로 함.
  - 중복되는 Identity 함수를 제거하고, 제거가능한 파이썬 연산(e.g., getitem, getattr)들을 최대한 제거하는 것을 목표로 함.
- [실행 및 평가 방법](docs/usage.md)

---

## 파이프라인

```
PyTorch Model → FX Graph → Gawee IR → JSON → MLIR(Gawee Dialect) → Linalg → SCF → LLVM
```

### 프론트엔드 (Python)
- torch fx 그래프를 Gawee IR로 파싱
- 사전 정의된 cost를 기반으로 베이스라인의 cost 측정
- Gawee IR에서 정의된 pass를 기반으로 최적화 수행
- IR을 JSON 형태로 저장

### 미들엔드 (C++ / MLIR)
- JSON 그래프를 MLIR Gawee Dialect으로 변환 (MLIREmitter)
- Gawee Dialect → Linalg Dialect 변환 (GaweeToLinalg)
- Linalg → Bufferization → SCF loops → LLVM Dialect까지 multi-stage lowering 수행
- `gawee-opt`, `gawee-translate` 두 개의 CLI 도구로 동작

---

## 프로젝트에서 다루는 핵심 개념

### 1. Gawee IR 설계

**그래프 분석과 최적화를 위해 필요한 정보만을 명시적으로 표현하는 자체 IR**를 정의.

- 연산 노드와 데이터 흐름의 명확한 분리
- shape / dtype / layout / data 정보의 명시적 표현
- Graph: Nodes (연산) / Values (텐서)
- Node: op type / input / output / attributes / fx Node
- Value: shape / dtype / producer / consumers / data(only for constant)

---

### 2. 그래프 분석

최적화를 위해 다음과 같은 분석을 수행.

- Shape inference
- Constant propagation
- Graph traversal (topological order)
- 연산 비용 추정:
  - FLOPs
  - 메모리 접근량(읽기/쓰기 추정)

---

### 3. 프론트엔드 최적화

프론트엔드에서는 **그래프 레벨 최적화**만 수행.

- Constant Folding
  - 상수 서브그래프를 컴파일 타임에 계산
- Operator Fusion
  - 연속된 연산 패턴을 하나의 fused operator로 결합
  - 예: Conv + BatchNorm, Conv + Add
- fx에 존재하는 파이썬 연산 제거
- [구현된 최적화 패스 설명](docs/arch.md)

---

### 4. MLIR Gawee Dialect 정의

**TableGen을 사용하여 딥러닝 연산에 대응하는 커스텀 MLIR Dialect**를 정의.

- `gawee` namespace 아래 Conv2D / ReLU / Add / BatchNorm / MaxPool / AdAvgPool / Flatten / Linear 연산 정의
- 각 연산의 입출력 타입, attribute(stride, padding, dilation 등)를 TableGen으로 선언
- TableGen에서 C++ boilerplate 코드 자동 생성

---

### 5. JSON → MLIR 변환 (MLIREmitter)

프론트엔드에서 출력한 **JSON 그래프를 MLIR Gawee Dialect 연산으로 변환**.

- JSON에서 입력, 출력, 가중치, 노드 정보를 파싱
- 가중치 텐서를 함수 인자로 등록
- 노드를 topological order로 순회하며 `gawee.*` 연산을 생성
- `func.func @main(...)` 형태의 MLIR 모듈을 출력

---

### 6. Gawee → Linalg Lowering

**OpConversionPattern을 사용하여 Gawee Dialect 연산을 Linalg Dialect으로 변환**.

- `gawee.conv` → `linalg.conv_2d_nchw_fchw` (padding 포함)
- `gawee.relu` → `linalg.generic` (max(x, 0) body)
- `gawee.add` → `linalg.add`
- BatchNorm / MaxPool / AdAvgPool / Flatten / Linear 각각에 대한 lowering 패턴 구현
- ConversionTarget, TypeConverter를 활용한 dialect 변환 프레임워크 사용

---

### 7. Multi-stage Lowering

Linalg에서 **LLVM IR까지 다단계 lowering 파이프라인**을 구성.

- `--convert-gawee-to-linalg`: Gawee → Linalg
- `--gawee-to-loops`: Gawee → Linalg → Bufferization → SCF loops
- `--gawee-to-llvm`: Gawee → Linalg → Bufferize → SCF → LLVM (전체 파이프라인)
- tensor → memref 변환(bufferization)을 포함

---

## 최적화 결과

### ResNet-18
```
Before: 69 nodes → After: 49 nodes
  - ConvBNFolding: 20회 적용
  - 메모리 읽기: 37.2MB → 27.3MB (26.7% 감소)
  - 메모리 쓰기: 32.9MB → 23.0MB (30.2% 감소)
```

### UNet
```
Before: 196 nodes → After: 116 nodes
  - IdentityElimination: 12회, ConvBNFolding: 46회, PythonOpElimination: 22회
  - 메모리 읽기: 136.6MB → 94.9MB (30.5% 감소)
  - 메모리 쓰기: 116.0MB → 83.7MB (27.9% 감소)
```

---

## 프로젝트 구조

```
gawee/
├── gawee_ir/                  # 프론트엔드 (Python)
│   ├── graph.py               #   Gawee IR 정의 (Graph / Node / Value)
│   ├── parser.py              #   PyTorch FX → Gawee IR 변환
│   ├── mapper.py              #   PyTorch 연산 → Gawee 연산 매핑
│   ├── translator.py          #   Gawee IR → JSON 변환
│   ├── analysis/              #   Shape inference, Cost 분석
│   └── passes/                #   최적화 패스 (Conv-BN folding 등)
├── middle/mlir/               # 미들엔드 (C++ / MLIR)
│   ├── include/Gawee/         #   TableGen 정의 (Dialect, Ops)
│   ├── lib/Gawee/             #   Dialect 등록
│   ├── lib/Conversion/        #   Gawee → Linalg 변환 패턴
│   ├── lib/Emit/              #   JSON → MLIR 변환 (MLIREmitter)
│   └── tools/                 #   gawee-opt, gawee-translate
├── scripts/                   # 실행 스크립트
├── jsondata/                  # 프론트엔드 출력 JSON
└── docs/                      # 문서
```

---

## 참고

- PyTorch fx 문서
- ONNX 공식 스펙
- TVM 아키텍처 문서
- MLIR 공식 문서 (Dialects, TableGen, Conversion)
