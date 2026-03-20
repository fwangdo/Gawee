# To Fix

현재 코드베이스는 "학습용으로 끝까지 구현해본 DL compiler prototype"으로서는 강점이 분명하지만, 정확성 검증과 end-to-end 완결성 측면에서는 보강이 필요하다. 아래 항목들은 코드 리뷰 기준으로 우선 정리해야 할 작업들이다.

## 목적

- 포트폴리오 관점:
  현재 프로젝트가 무엇을 실제로 지원하는지와 무엇이 아직 미완성인지 명확히 구분한다.
- 엔지니어링 관점:
  "동작해 보이는 데모"가 아니라, 재현 가능하고 검증 가능한 컴파일러 파이프라인으로 끌어올린다.
- 상업화 관점:
  향후 독립 제품으로 발전시키기 전에 correctness, testability, contract clarity를 먼저 확보한다.

## 핵심 판단

- 이 프로젝트는 방향이 좋다.
  `PyTorch FX -> custom IR -> rewrite passes -> JSON -> MLIR dialect -> Linalg -> LLVM` 구조는 충분히 포트폴리오 가치가 있다.
- 다만 현재 완성도는 "강한 프로토타입"에 가깝다.
  상업용 소프트웨어로 이어가려면 correctness 보장과 지원 범위 명세가 먼저 필요하다.
- 따라서 `progress.md`를 계속 진행하는 것은 가능하지만, 아래 `P0` 항목은 너무 늦게 미루면 안 된다.

## P0. 가장 먼저 고칠 것

### 1. JSON -> MLIR 경로에서 graph constants를 실제로 처리하기

문제:
- frontend는 rewrite 과정에서 생성된 상수를 `values`와 `constants/*.bin`으로 export한다.
- 하지만 MLIR emitter는 입력 텐서와 일부 weight/bias만 `valueMap`에 등록한다.
- 따라서 pass가 만든 상수를 입력으로 받는 node는 middle-end에서 lookup에 실패한다.

영향:
- `PythonOpElimination`, `ConstantFolding`이 만든 결과가 end-to-end lowering에서 깨질 수 있다.
- 현재 파이프라인은 "상수 rewrite가 없는 일부 경로"에만 의존해 돌아갈 가능성이 있다.

확인 위치:
- [translator.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/translator.py#L130)
- [translator.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/translator.py#L175)
- [MLIREmitter.cpp](/Users/hdy/code/portfolio/Gawee/middle/mlir/lib/Emit/MLIREmitter.cpp#L50)
- [MLIREmitter.cpp](/Users/hdy/code/portfolio/Gawee/middle/mlir/lib/Emit/MLIREmitter.cpp#L137)
- [MLIREmitter.cpp](/Users/hdy/code/portfolio/Gawee/middle/mlir/lib/Emit/MLIREmitter.cpp#L360)

해야 할 일:
- JSON `values`의 constant 항목을 emitter가 읽을 수 있게 설계 정리
- 방법 결정:
  1. constants를 함수 인자로 올릴지
  2. emitter 내부에서 dense constant op로 materialize할지
- `lookupValue()` 전에 constants도 `valueMap`에 등록되게 수정
- constant가 포함된 그래프를 end-to-end로 검증하는 테스트 추가

권장:
- 학습 목적과 상업화 가능성을 같이 보려면, "현재는 constant를 함수 인자로 모델링한다"처럼 단일 규약으로 통일하는 편이 낫다.

### 2. 테스트 체계 복구

문제:
- `pytest -q` 실행 시 `scripts/test_lowering.py`와 `scripts/test_shape.py`의 `test_model(...)`이 pytest test 함수로 오인되어 fixture error가 발생한다.
- 핵심 회귀 테스트 파일인 `tests/test_passes.py`, `tests/test_equivalence.py`는 비어 있다.

직접 확인 결과:
- `pytest -q` 결과: `1 passed, 2 errors`

영향:
- correctness를 증명할 수 없다.
- README의 최적화 결과나 지원 여부를 신뢰하기 어렵다.

확인 위치:
- [scripts/test_lowering.py](/Users/hdy/code/portfolio/Gawee/scripts/test_lowering.py#L117)
- [scripts/test_shape.py](/Users/hdy/code/portfolio/Gawee/scripts/test_shape.py#L60)
- [tests/test_passes.py](/Users/hdy/code/portfolio/Gawee/tests/test_passes.py#L1)
- [tests/test_equivalence.py](/Users/hdy/code/portfolio/Gawee/tests/test_equivalence.py#L1)

해야 할 일:
- `scripts/` 아래 파일은 pytest 수집 대상에서 제외하거나 함수명을 변경
- `tests/` 아래에 실제 unit/integration/regression test 작성
- 최소 기준:
  - parser smoke test
  - pass 단위 테스트
  - translator/export validation test
  - end-to-end semantic equivalence test

### 3. 문서와 실제 지원 범위 일치시키기

문제:
- README는 일반적인 DL compiler 파이프라인처럼 읽히지만, 실제 구현은 특정 모델 경로에 강하게 최적화되어 있다.
- 특히 BatchNorm은 frontend folding 전제에 의존하고, emitter dispatch는 제한된 op만 직접 지원한다.

영향:
- 포트폴리오에서 과장으로 보일 수 있다.
- 면접에서 "정말 BN lowering도 끝까지 되나?" 같은 질문이 나오면 방어가 어려워진다.

확인 위치:
- [README.md](/Users/hdy/code/portfolio/Gawee/README.md#L82)
- [README.md](/Users/hdy/code/portfolio/Gawee/README.md#L103)
- [MLIREmitter.cpp](/Users/hdy/code/portfolio/Gawee/middle/mlir/lib/Emit/MLIREmitter.cpp#L185)

해야 할 일:
- README에 현재 지원 범위를 명시
- "general-purpose compiler"처럼 읽히는 표현은 줄이고, "ResNet/UNet-oriented prototype compiler"로 정확히 표현
- frontend에서 folding된 op와 middle-end가 직접 lowering하는 op를 구분해서 문서화

## P1. correctness와 데이터 신뢰도 보강

### 4. pass 통계 누적 버그 수정

문제:
- `deleted_node`가 class state라 반복 실행 시 누적될 수 있다.
- `Passer.run()`은 pass 실행 전에 통계를 reset하지 않는다.

영향:
- 실험 결과가 과대 집계될 수 있다.
- 문서에 적힌 최적화 횟수의 신뢰도가 떨어진다.

확인 위치:
- [folder.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/folder.py#L13)
- [passer.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/passer.py#L31)
- [conv_bn_folding.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/conv_bn_folding.py#L155)

해야 할 일:
- `Passer.run()` 시작 시 각 pass counter reset
- 가능하면 class variable 대신 run-local result 반환 구조로 변경
- benchmark script가 여러 모델을 돌려도 독립된 결과를 내도록 수정

### 5. ConstantFolding의 문서/구현 불일치 수정

문제:
- `RESHAPE`, `REDUCE_MEAN` folding은 문서상 언급되지만 실제로는 `NotImplementedError`를 던진다.

영향:
- 특정 입력에서 pass 전체가 실패할 수 있다.
- "지원"과 "미지원" 경계가 불명확하다.

확인 위치:
- [constant_folding.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/constant_folding.py#L59)
- [constant_folding.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/constant_folding.py#L102)
- [README.md](/Users/hdy/code/portfolio/Gawee/README.md#L74)

해야 할 일:
- 둘 중 하나를 선택
  1. 실제 구현
  2. 미지원으로 명시하고 graceful skip 처리
- 학습 단계라면 우선 graceful skip이 낫다.

### 6. unsupported case를 fail-fast와 recoverable case로 분리

문제:
- 현재 여러 부분에서 `Exception`, `NotImplementedError`, string fallback이 섞여 있다.
- 일부는 개발 중 디버깅에는 편하지만, 사용자-facing compiler로 가기에는 계약이 불명확하다.

예시:
- parser unsupported FX op
- Python op resolution 실패
- translator attr serialization의 string fallback

확인 위치:
- [parser.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/parser.py#L209)
- [constant_folding.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/passes/constant_folding.py#L103)
- [translator.py](/Users/hdy/code/portfolio/Gawee/gawee_ir/translator.py#L103)

해야 할 일:
- unsupported op는 명시적인 에러 타입 사용
- recoverable case는 warning + skip으로 통일
- "왜 실패했는지"를 최종 사용자와 개발자가 모두 이해할 수 있게 메시지 구조화

## P2. 제품화 관점에서 이후 보강할 것

### 7. semantic equivalence를 실제 숫자로 검증

필수 질문:
- optimization 전후 출력이 같은가
- JSON export/import 후 의미가 보존되는가
- MLIR lowering 후 실행 결과가 reference와 일치하는가

해야 할 일:
- 작은 모델 2~3개에 대해 golden test 작성
- tolerance 기반 비교 도입
- Conv-BN folding, Identity elimination, Python op elimination에 대해 각각 개별 equivalence test 작성

### 8. 지원 모델과 비지원 모델 경계 명시

현재 관찰:
- ResNet 경로는 꽤 정리되어 있다.
- UNet은 `cat`, `interpolate` 확장이 진행 중이다.
- Dropout 등 일부 op는 여전히 스킵/미지원이다.

해야 할 일:
- "지원 모델"
- "부분 지원"
- "미지원"
세 범주로 README와 docs를 정리

### 9. IR contract 문서화

현재 필요한 것:
- Value가 constant일 때의 의미
- parameter/buffer/activation 구분
- JSON schema
- op attribute naming convention
- dtype/layout 가정

이유:
- frontend와 middle-end를 따로 고치기 시작하면 contract 문서가 없을 때 빠르게 깨진다.

## progress.md와의 관계

질문:
- 지금 `progress.md`를 수행하면서 익히는 중인데, 위 작업을 다 끝내고 `to_fix`를 수행해도 되는가?

답:
- 전부 다 끝낸 뒤 한꺼번에 하는 것은 비추천이다.
- 다만 `progress.md` 자체를 멈출 필요는 없다.

권장 순서:
1. `progress.md`의 기능 확장은 계속 진행
2. 하지만 아래 3개는 중간 시점에 반드시 먼저 반영
3. 그 후 나머지 `to_fix`를 정리

중간에 먼저 처리할 최소 항목:
- 테스트 체계 복구
- constants 처리 계약 정리
- README 지원 범위 정정

이유:
- 새 op를 더 추가해도 검증 기반이 없으면 "지원하는 것처럼 보이는 코드"만 늘어난다.
- 지금 진행 중인 `cat`, `interpolate`도 결국 correctness를 증명해야 포트폴리오 가치가 생긴다.
- 따라서 `progress.md`와 `to_fix`는 순차 작업이 아니라, 일부는 병행해야 한다.

## 실전 권장 운영 방식

가장 현실적인 방식은 아래다.

### 트랙 A. 계속 학습하며 구현

- `middle/mlir/docs/progress.md`대로 `cat`, `interpolate`를 진행
- MLIR dialect / emitter / lowering 감각을 계속 익힘

### 트랙 B. 품질 바닥 다지기

- pytest 수집 오류 먼저 수정
- `tests/`에 실제 회귀 테스트 추가
- constants end-to-end 처리 방식 확정
- README 표현 정리

이 두 트랙을 병행하면 된다. 완전히 분리해서 뒤로 미루는 것보다 낫다.

## 추천 우선순위

### 이번 주에 할 것

- pytest 수집 오류 수정
- `tests/test_passes.py` 작성
- `tests/test_equivalence.py` 작성
- constants 처리 방식 결정
- README에 현재 지원 범위 명시

### 그 다음

- `cat`, `interpolate` 구현 계속
- UNet full pipeline 연결
- benchmark/result 문서 재검증

### 상업화 전에는 반드시

- semantic equivalence 자동화
- unsupported op 정책 정리
- JSON/IR contract 문서화
- 실험 결과 재현 스크립트 정리

## 결론

- `progress.md`를 수행하는 것은 계속해도 된다.
- 하지만 `to_fix`를 전부 뒤로 미루는 것은 좋지 않다.
- 최소한 `P0`의 일부는 지금 진행 중인 학습과 병행해야 한다.
- 포트폴리오 설득력은 "기능 개수"보다 "검증된 기능"에서 나온다.
