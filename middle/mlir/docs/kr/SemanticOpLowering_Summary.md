# Semantic Op Lowering Summary

## 왜 이 문서가 필요한가

이번 변경의 핵심은 "지원 op를 늘린다"가 아니라, `front`에서 과하게 rewrite 하던 책임 일부를 `middle/mlir`로 옮기는 것이다.

특히 LLM 계열 모델에서는 다음 문제가 컸다.

- `Gather`, `Range` 같은 op가 자주 등장한다.
- 이 op들을 front에서 다른 op 조합으로 억지 변환하면 shape/axis 의미가 흐려진다.
- 나중에 fallback 여부를 판단할 경계도 사라진다.

그래서 기준을 다음처럼 잡는다.

## 기준 1: semantic op는 gawee에 남긴다

다음 op는 단순 산술이 아니라 shape/axis/lookup 의미를 가진다.

- `gawee.gather`
- `gawee.gather_elements`
- `gawee.range`
- `gawee.resize`
- `gawee.split`
- `gawee.tile`

이런 op는 ONNX emitter가 바로 `linalg.generic`로 쪼개지 않고, 먼저 `gawee` op로 만든다.

이렇게 해야 좋은 점은 다음과 같다.

- ONNX 의미가 middle-end 경계까지 보존된다.
- 나중에 느린 lowering만 선택적으로 fallback 하기가 쉽다.
- front rewrite를 줄여도 unsupported op가 남지 않는다.

## 기준 2: trivial decomposition은 직접 낮춘다

모든 op를 새 dialect op로 만들 필요는 없다.

예를 들어 다음 op는 의미가 비교적 단순하다.

- `Pow`
- `Neg`
- `Sin`
- `Cos`
- `And`
- `LessOrEqual`
- `IsNaN`
- `Mod`
- `Constant`
- `ConstantOfShape`

이런 op는 emitter에서 바로 `tensor/linalg/math/arith` 조합으로 내리는 편이 낫다.

이유는 다음과 같다.

- dialect가 불필요하게 비대해지지 않는다.
- lowering 패턴 수가 줄어든다.
- 성능/semantic 측면에서 따로 fallback 경계를 둘 가치가 작다.

## 이번 파이프라인에서 각 단계가 하는 일

### 1. TD

`GaweeOps.td`에 semantic op를 선언한다.

예:

- `GatherOp(data, indices, axis)`
- `RangeOp(start, limit, delta)`
- `SplitOp(input, splitSizes, axis) -> variadic results`

TD 단계에서 중요한 것은 "어떤 정보가 lowering에 필요하냐"다.

예를 들어:

- `Gather`는 `axis`가 필요하다.
- `Split`은 `axis`와 `splitSizes`가 필요하다.
- `Resize`는 최소한 `mode`, `coordinate_transformation_mode`, `nearest_mode`가 필요하다.

## 2. Emission

ONNX emitter는 다음 책임만 진다.

- operand lookup
- result type lookup
- ONNX attr normalization
- `gawee.*` op 생성

중요한 점:

- emitter는 가능한 한 ONNX 의미를 보존해야 한다.
- 복잡한 indexing loop는 lowering으로 보내는 편이 낫다.

예:

- `emitGather`는 `gawee.gather`를 만든다.
- `emitRange`는 `gawee.range`를 만든다.
- `emitSplit`는 variadic result를 갖는 `gawee.split`를 만든다.

## 3. Linalg Lowering

여기서 실제 decomposition이 일어난다.

### Gather

`tensor.generate` 안에서 output index를 돌며:

1. gather axis 전의 index는 그대로 쓴다.
2. axis 위치 index는 `indices`에서 읽는다.
3. axis 뒤 index는 ONNX 결과 shape 규칙에 맞게 다시 매핑한다.

### GatherElements

output과 `indices` shape가 같으므로, output position을 그대로 돌면서:

1. 현재 position의 `indices` 값을 읽는다.
2. 그 값만 axis 위치에 대입해서 `data`를 읽는다.

### Range

1-D output tensor를 `tensor.generate`로 만든다.

각 원소 `i`는:

- `start + i * delta`

dynamic length가 있으면 lowering에서 `start/limit/delta`로 길이를 계산해야 한다.

이번 구현은 `delta`가 양의 constant integer일 때 dynamic length를 계산한다.

### Split

`tensor.extract_slice`를 여러 번 만들어 variadic result로 바꾼다.

핵심은:

- 각 output의 offset을 누적한다.
- axis 방향 size만 split size로 바뀐다.

### Tile / Resize

둘 다 현재는 correctness 우선 구현이다.

- `Tile`: modulo indexing
- `Resize`: nearest/asymmetric/floor + integer scale만 허용

즉, 느릴 수는 있지만 unsupported op는 아니다.

## re-implement 할 때 꼭 이해해야 할 것

### 1. "semantic 보존"과 "decomposition 위치"를 구분하라

질문:

- 이 op는 dialect 경계에서 의미를 보존할 가치가 있는가?
- 아니면 바로 generic loop로 쪼개도 되는가?

이 판단이 먼저다.

### 2. ONNX shape 규칙을 lowering으로 옮길 수 있어야 한다

예:

- `Gather` output rank 계산
- `Split` output size 배분
- `Range` 동적 길이 계산

### 3. variadic result op를 무서워하지 말 것

`Split`처럼 output이 여러 개인 op는 TD와 lowering에서 자연스럽게 표현할 수 있다.

핵심은:

- emitter에서 result type 목록을 준비하고
- lowering에서 `SmallVector<Value>`로 바꿔 `replaceOp` 하는 것

## 스스로 다시 구현할 때 순서

1. ONNX spec에서 op 의미와 shape rule을 적는다.
2. TD에 lowering에 필요한 operand/attr/result만 선언한다.
3. emitter는 "읽고 생성"만 한다.
4. lowering에서 `tensor.generate` / `extract_slice` / `linalg.generic`로 푼다.
5. 실제 모델 하나로 번역과 `gawee-to-llvm`를 확인한다.

이 순서를 지키면 front rewrite를 덜어내면서도 op coverage를 안정적으로 늘릴 수 있다.
