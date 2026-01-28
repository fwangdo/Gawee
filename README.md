# gawee

`gawee`는 PyTorch로 작성된 딥러닝 모델을 자체 IR로 변환한 뒤, 그래프 분석 및 최적화(graph optimization)를 수행하는 딥러닝 컴파일러 프론트엔드 프로젝트입니다.
본 프로젝트의 목적은 실무 딥러닝 컴파일러(예: TVM 계열)가 실제로 수행하는 문제를 최소한의 스코프로 직접 설계·구현·검증하는 것입니다.

---

## 대상 모델

- **ResNet-18**
  - ImageNet용 표준 ResNet-18 구조를 기반으로 함
  - Conv / BatchNorm을 fusion하여 그래프 노드를 줄이는 것을 목표로 함.

- **Unet**
  - ImageNet용 표준 Unet 구조를 기반으로 함
  - 중복되는 Identity 함수를 제거하고, 제거가능한 파이썬 연산(e.g., getitem, getattr)들을 최대한 제거하는 것을 목표로 함. 

---

## 프로젝트에서 다루는 핵심 개념

### 1. gawee IR 설계

**그래프 분석과 최적화를 위해 필요한 정보만을 명시적으로 표현하는 자체 IR**를 정의.

IR 설계의 주요 관심사:

- 연산 노드와 데이터 흐름의 명확한 분리
- shape / dtype / layout / data 정보의 명시적 표현
- 그래프 순회 및 rewrite가 용이한 구조

예시 개념 구조:

- Graph
  - Nodes (연산)
  - Values (텐서)
- Node
  - op type
  - input / output
  - attributes
  - fx Node 
- Value
  - shape
  - dtype
  - producer / consumers
  - data(only for constant)

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

### 3. 그래프 최적화

본 프로젝트에서는 **그래프 레벨 최적화**에 집중한다.

구현 예정 최적화 예시:

- Constant Folding
  - 상수 서브그래프를 컴파일 타임에 계산
- Operator Fusion
  - 연속된 연산 패턴을 하나의 fused operator로 결합
  - 예: Conv + BatchNorm, Conv + Add 
- fx에 존재하는 파이썬 연산 제거
- 기타 그래프 단순화(canonicalization)

각 최적화는 onnx basic graph optimizations를 참조.
- https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#basic-graph-optimizations

---

## 파이프라인

### 프론트엔드 
- torch fx 그래프를 Gawee ir로 파싱
- 사전 정의된 cost를 기반으로 베이스라인의 cost 측정
- Gawee ir에서 정의된 pass를 기반으로 최적화 수행
- ir을 json 형태로 저장

### 미들엔드
- mlir 기반 수행 기능 정의(TODO)

---

## 참고

- PyTorch fx 문서
- ONNX 공식 스펙
- TVM 아키텍처 문서

---

## 사용 방법 및 실험 결과

- [실행 방법](docs/usage.md)
- [구현된 최적화 패스 설명](docs/arch.md)