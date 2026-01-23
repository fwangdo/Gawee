# gawee

`gawee`는 PyTorch로 작성된 딥러닝 모델을 자체 IR로 변환한 뒤, 그래프 분석 및 최적화(graph optimization)를 수행하는 딥러닝 컴파일러 프론트엔드 프로젝트입니다.
본 프로젝트의 목적은 실무 딥러닝 컴파일러(예: TVM 계열)가 실제로 수행하는 문제를 최소한의 스코프로 직접 설계·구현·검증하는 것입니다.

---

## 프로젝트 목표

- PyTorch 모델을 **정적 계산 그래프**로 변환하는 프론트엔드 파이프라인 이해
- 그래프 레벨 최적화의 원리와 정합성(soundness) 이해 및 구현
- 최적화 전/후 그래프의 **FLOPs 및 메모리 접근 연산 비교를 통한 효과 검증**

---

## 전체 파이프라인

PyTorch (ResNet)
→ gawee IR 변환
→ 그래프 분석
→ 그래프 최적화
→ 최적화 전/후 비교

---

## 대상 모델

- **ResNet-18 (최소 구성)**
  - ImageNet용 표준 ResNet-18 구조를 기반으로 함
  - 목적은 모델 정확도가 아니라 **그래프 구조와 연산 패턴**
  - Conv / BatchNorm / ReLU / Residual connection 등
    그래프 최적화 논의에 충분한 연산을 포함

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
- Value
  - shape
  - dtype
  - producer / consumers

---

### 2. 그래프 분석

최적화를 위해 다음과 같은 분석을 수행한다.

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
  - 예: Conv + BatchNorm + ReLU
- 불필요한 reshape / transpose 제거 또는 정규화
- 그래프 단순화(canonicalization)

각 최적화는 onnx basic graph optimizations 를 기초로 한다:

- https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html#basic-graph-optimizations

---

### 4. 정합성 검증

최적화는 반드시 의미 보존이어야 하므로,  
최적화 전/후 그래프에 대해 다음을 검증한다.

- shape 일치 여부 확인
- dtype 일치 여부 확인

---

### 5. 성능 비교

최적화 효과는 다음 두 지표로 평가한다.

1. **FLOPs 비교**
   - 이론적 연산량 감소 확인
2. **Runtime 비교**

이를 통해:
- “왜 이 최적화가 효과적인가”
- “어떤 비용이 줄어들었는가”
를 정량적으로 설명하는 것을 목표로 한다.

---

## 이 프로젝트가 보여주고자 하는 것

- 딥러닝 컴파일러 프론트엔드가 해결하는 **실제 문제의 구조**
- IR 설계가 분석과 최적화에 미치는 영향
- 그래프 최적화가 단순한 트릭이 아니라  
  **전제 조건과 정합성 판단이 필요한 컴파일러 문제**임을 이해하고 있음을 증명

---

## 파이프라인

- torch fx 그래프를 Gawee ir로 파싱
- Gawee ir에서 최적화 수행
- ir을 json 형태로 저장
- mlir 기반 수행 기능 정의(TODO)

---

## 참고

- PyTorch ONNX Export 문서
- ONNX 공식 스펙
- TVM 아키텍처 문서