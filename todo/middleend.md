# TODO: Gawee Middle-End Challenges

이 문서는 `Gawee`의 middle-end에서
단순히 ``구현을 읽고 설명할 수 있다''를 넘어서,
`실제 엔지니어링 문제를 정의하고 해결해 본 흔적`을 남기기 위한 TODO다.

핵심 목표는 아래다.

> 나는 middle-end가 어떤 층위의 문제를 푸는지 알고 있고,
> 실제 imported graph의 불안정성, unsupported op, brittle pass, invariant 문제를
> 정의하고 해결책을 설계해 본 경험이 있다.

즉 이 문서는 ``pass를 하나 더 만든다''가 아니라
`어떤 middle-end challenge를 풀어야 진짜 문제를 풀어본 것처럼 보이는가`를 정리한다.

---

## 1. 목표 인상

이 문서를 끝까지 수행했을 때 남아야 하는 인상은 아래다.

- imported graph는 생각보다 messy하다는 점을 안다.
- 같은 의미라도 graph 표현이 제각각이라 middle-end가 canonicalization을 해야 한다는 점을 안다.
- backend가 못 받는 고수준 op를 primitive subset으로 lowering하는 문제를 안다.
- rewrite를 추가할수록 verifier와 invariant가 중요해진다는 점을 안다.
- fusion은 ``좋아 보이는 rewrite''가 아니라 legality와 profitability를 함께 봐야 한다는 점을 안다.

즉 `Gawee`는 toy compiler가 아니라,
`실제 graph를 받아 middle-end issue를 처리하는 prototype`처럼 보여야 한다.

---

## 2. 지금 Gawee middle-end의 상태

현재 `Gawee`의 강점은 분명하다.

- graph parsing이 있다.
- `Value / Node / Graph` IR가 있다.
- canonicalization과 elimination pass가 일부 있다.
- `Conv-BN`, `Conv-Add` 같은 legal rewrite가 있다.
- MLIR/linalg로 이어지는 lowering path가 있다.

하지만 지금 상태만으로는 아래 질문에 답하기 약하다.

- imported graph가 왜 pass-friendly하지 않은가
- unsupported op를 어떻게 target-friendly form으로 바꾸는가
- rewrite가 늘어날수록 graph consistency를 어떻게 보장하는가
- 왜 어떤 fusion은 하고 어떤 fusion은 하지 않는가

즉 현재는 ``구현한 것''은 있으나,
`실제 challenge를 만나 해결했다`는 서사가 약하다.

---

## 3. 가장 추천하는 middle-end 문제들

아래 네 문제는 모두 middle-end답고,
현재 `Gawee` 위에서 충분히 reasoning 중심으로 발전시킬 수 있다.

---

## Challenge A. Shape-Aware Canonicalization

### 문제 상황

실제 imported graph는 같은 의미라도 다양한 형태를 가진다.
예를 들어 아래는 모두 흔한 문제다.

- `squeeze -> unsqueeze`가 쓸모없이 남아 있다
- `reshape`가 실제로는 no-op인데 graph에 남아 있다
- `transpose -> transpose`가 서로 상쇄되는데 cleanup이 안 된다
- `expand`가 필요한 경우와 불필요한 경우가 섞여 있다

이 상태에서는 후속 pattern match가 brittle해진다.
즉 optimization 문제 이전에,
`graph 표현 다양성` 자체가 문제다.

### 해결 방향

`shape-preserving canonicalization pass`를 만든다.

예시:

- [ ] no-op reshape 제거
- [ ] redundant squeeze/unsqueeze 제거
- [ ] transpose pair elimination 일반화
- [ ] shape-only op chain을 canonical form으로 정리
- [ ] 필요한 expand와 제거 가능한 expand를 구분

### 왜 middle-end다운 문제인가

이건 backend처럼 tile을 고르는 문제가 아니라,
후속 pass가 더 안정적으로 먹히도록
graph 표현을 정리하는 문제다.
정확히 middle-end의 전형적인 일이다.

### 면접에서 이렇게 말할 수 있어야 한다

> 실제 challenge는 optimization 자체보다 imported graph가 pass-friendly하지 않다는 점이었다.
> 그래서 같은 의미의 shape manipulation들을 canonical form으로 모으는 pass를 설계했고,
> 그 결과 후속 pattern match의 안정성이 좋아졌다.

### TODO

- [ ] 현재 graph에서 shape-only op family를 분류
- [ ] `no-op reshape` 판별 조건 정의
- [ ] `squeeze/unsqueeze` cancel 조건 정의
- [ ] `transpose pair` 일반 상쇄 규칙 정의
- [ ] shape canonicalization 전/후 diff 리포트 작성
- [ ] pattern match 성공률 변화 관찰

---

## Challenge B. Unsupported Op Lowering To Primitive Subset

### 문제 상황

실제 target backend는 고수준 op를 전부 지원하지 않는다.
특히 transformer 계열에서는 아래가 대표적이다.

- `LayerNorm`
- `SkipLayerNorm`
- `GELU`
- `Where`-based attention mask
- generic `Gather`

이 문제는 ``모델이 안 돌아간다''의 원인 중 상당 부분을 차지한다.

### 해결 방향

지원 가능한 primitive subset을 정하고,
unsupported op를 그 subset으로 lowering한다.

예시:

- [ ] `LayerNorm -> ReduceMean/Sub/Mul/ReduceMean/Add/Sqrt/Div/Mul/Add`
- [ ] `SkipLayerNorm -> Add + LayerNorm`
- [ ] `GELU -> tanh-based path`
- [ ] `Where mask -> additive mask`
- [ ] `Gather -> axis=0 only equivalent form` 가능 조건 정리

### 왜 middle-end다운 문제인가

이건 target이 못 받는 high-level semantics를
더 explicit하고 target-friendly한 op sequence로 바꾸는 작업이다.
즉 graph optimization과 backend codegen 사이의 전형적인 middle-end 문제다.

### 면접에서 이렇게 말할 수 있어야 한다

> 실제 challenge는 high-level ONNX op를 target이 직접 못 받는다는 점이었다.
> 그래서 primitive subset을 먼저 정의하고,
> unsupported op를 semantic contract를 보존하는 방식으로 decomposition/lowering하는 pass를 설계했다.

### TODO

- [ ] supported primitive op set 명시
- [ ] model별 unsupported op audit 자동화
- [ ] `LayerNorm`, `GELU`, `mask` lowering 우선순위 확정
- [ ] 각 lowering의 legality 조건 정리
- [ ] tolerance / validation policy 문서화
- [ ] skip reason schema 정의

---

## Challenge C. IR Verifier And Pass Regression Harness

### 문제 상황

rewrite pass가 늘어날수록
``graph가 여전히 sane한가''를 보장하기 어려워진다.

대표적으로 이런 문제가 생긴다.

- `Value`의 producer/consumer 연결이 깨진다
- graph output이 dangling value를 가리킨다
- dead node가 남는다
- constant/activation 구분이 흐려진다
- shape metadata가 pass 후 일관되지 않다

### 해결 방향

각 pass 뒤에 돌릴 수 있는 `IR verifier`를 만든다.
이 verifier는 optimization을 하지 않고,
`graph invariant가 유지되는지`만 검사한다.

예시 invariant:

- [ ] 각 `Value`의 producer가 명확한가
- [ ] dead consumer reference가 없는가
- [ ] graph output은 유효한 value인가
- [ ] constant value는 unexpected producer를 갖지 않는가
- [ ] node input/output 연결이 일관적인가

### 왜 middle-end다운 문제인가

현업 middle-end는 ``pass 하나 더 추가''보다
`pass를 많이 넣어도 안 깨지게 하는 인프라`가 중요할 때가 많다.
verifier는 그 핵심이다.

### 면접에서 이렇게 말할 수 있어야 한다

> rewrite가 늘어날수록 correctness 문제는 연산 의미보다 graph consistency에서 더 자주 터졌다.
> 그래서 pass 뒤에 invariant를 점검하는 verifier를 넣어 regression을 막는 쪽으로 접근했다.

### TODO

- [ ] `Gawee IR invariant`를 문장으로 고정
- [ ] verifier 체크리스트 작성
- [ ] pass 실행 전/후 verifier hook 추가
- [ ] 실패 시 human-readable report 작성
- [ ] regression test 최소 3개 추가

---

## Challenge D. Profitability-Aware Fusion

### 문제 상황

legal하다고 해서 항상 fuse하는 것은 위험하다.
예를 들어 아래 경우는 생각해 볼 수 있다.

- intermediate가 multi-consumer다
- broadcast semantics가 애매하다
- backend가 이미 더 잘 fuse한다
- fused form이 오히려 후속 lowering을 어렵게 한다

즉 ``fuse 가능''과 ``fuse 해야 함''은 다르다.

### 해결 방향

현재 `Conv-BN`, `Conv-Add` 수준을 넘어서
``언제 fuse하지 않을 것인가''를 명시한 profitability guard를 둔다.

예시:

- [ ] multi-consumer intermediate면 skip
- [ ] shape/broadcast ambiguity면 skip
- [ ] backend에서 이미 common pattern이면 skip 근거 남기기
- [ ] pass log에 legal-but-skipped 구분 남기기

### 왜 middle-end다운 문제인가

middle-end는 ``의미를 보존하는 rewrite를 만든다''에서 멈추지 않고,
`언제 적용하는 것이 pipeline 전체에 유리한가`까지 본다.
그게 profitability reasoning이다.

### 면접에서 이렇게 말할 수 있어야 한다

> fusion을 무조건 많이 넣는 방향보다는,
> legality와 별도로 profitability guard를 두어
> skip reason이 남는 pass를 만드는 쪽이 더 엔지니어링스럽다고 판단했다.

### TODO

- [ ] 현재 fusion pass의 side condition 정리
- [ ] multi-consumer intermediate 탐지 추가
- [ ] legal-but-not-profitable 사례 2개 정리
- [ ] skip reason을 report에 남기기

---

## 4. 우선순위

시간 대비 효과를 생각하면 아래 순서가 좋다.

1. `Unsupported Op Lowering`
2. `Shape-Aware Canonicalization`
3. `IR Verifier`
4. `Profitability-Aware Fusion`

이 순서가 좋은 이유는,
첫 두 개가 가장 `JD 친화적`이고,
세 번째가 `엔지니어링 깊이`,
네 번째가 `reasoning maturity`를 보여주기 때문이다.

---

## 5. 최소 성공 조건

이 문서를 다 하려는 것이 목표는 아니다.
최소한 아래 셋 중 하나만 제대로 해도
``실제 challenge를 풀어봤다''는 말이 성립한다.

- [ ] unsupported op lowering 하나를 끝까지 구현하고 validation까지 붙인다
- [ ] shape canonicalization pass를 만들고 후속 pattern match 안정성 개선을 보여 준다
- [ ] IR verifier를 넣고 pass regression case를 실제로 잡아낸다

---

## 6. 면접용 한 문장 포지셔닝

이 TODO를 실제로 수행한 뒤에는 아래 문장을 말할 수 있어야 한다.

> 저는 graph parsing과 lowering을 구현한 수준에서 멈추지 않고,
> imported graph의 shape noise를 canonicalize하고,
> target이 못 받는 op를 primitive subset으로 lowering하며,
> 그 과정이 안 깨지게 verifier로 묶는 middle-end 문제를 실제로 다뤄봤습니다.

이 문장이 나오면
``구현을 안다''에서
`엔지니어처럼 문제를 풀어봤다`로 한 단계 올라간다.
