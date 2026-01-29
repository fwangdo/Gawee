## pass 구조

`passes/` 디렉터리는 아래와 같이 구성됩니다. 각 파일은 “하나의 pass(또는 pass 실행기)”에 대응하며, 목표는 변환을 통한 그래프를 단순화 및 연산 감소입니다.

```text
passes/
├── canonicalize.py
├── constant_folding.py
├── conv_add_folding.py
├── conv_bn_folding.py
├── elim_identity.py
├── errors.py
├── folder.py
└── passer.py
```

---

### folder.py

* 모든 pass가 공유하는 공통 인터페이스를 제공하는 클래스입니다.
* 각 pass는 이 클래스를 상속해 `run(graph)` 형태로 구현됩니다.

### passer.py

* 여러 pass를 실행하는 실행기입니다.
* pass 적용 전/후의 통계를 비교할 수 있도록, 실행 결과(예: 노드 수 변화)를 취합하는 역할을 합니다.

### errors.py

* pass 수행 중 발생할 수 있는 예외/오류를 정리한 파일입니다.
* (예: unsupported case, 그래프 불일치 등) 상황을 명확히 구분해 디버깅을 돕습니다.

---

### canonicalize.py

* FX 그래프에서 흔히 나오는 **Python bookkeeping 노드**(예: `getattr`, `getitem` 등)를 정리하여 IR을 더 “연산 그래프”답게 만드는 pass입니다.
* 핵심은 실제 텐서 계산이 아닌 데이터 접근을 상수로 변경하여 불필요한 중간 노드를 제거하는 것입니다.

### elim_identity.py

* Identity 연산 노드를 제거하는 단순 정리 pass입니다.
* 그래프의 dataflow를 유지하면서, Identity 출력 사용처를 입력으로 치환한 뒤 노드를 삭제합니다.

### constant_folding.py

* 입력이 모두 상수로 결정되는 subgraph를 컴파일 타임에 계산해 상수로 치환하는 pass입니다.
* 런타임 계산을 줄이고, 이후 pass(예: fuse/elim)의 패턴 매칭이 쉬워지도록 그래프를 단순화합니다.

---

## conv_bn_folding.py (Conv–BatchNorm Folding)

이 pass는 inference/eval 모드에서 자주 쓰는 최적화로, 다음 패턴을:

```
y = BN( Conv(x; W, b) )
```
BatchNorm을 제거하고, Conv의 파라미터를 재계산한 Conv’로 치환합니다:
```
y = Conv'(x; W', b')
```

즉, 그래프에서는 `Conv -> BN` 두 노드를 `Conv'` 하나로 줄입니다. 

### 1) BatchNorm의 inference 수식

Conv 출력(채널별)을 `u`라고 하면 BN은 채널 `c`에 대해:

* `gamma[c]`, `beta[c]` : 학습된 파라미터(affine)
* `mu[c]`, `var[c]` : running mean/var (buffer)
* `eps` : 안정화 상수(엡실론)

BN(inference)은:

[
BN(u_c) = \gamma_c \cdot \frac{u_c - \mu_c}{\sqrt{\mathrm{var}_c + \epsilon}} + \beta_c
]

여기서 채널별 상수:

[
a_c = \frac{\gamma_c}{\sqrt{\mathrm{var}_c + \epsilon}}
]

로 두면:

[
BN(u_c) = a_c \cdot u_c + (\beta_c - a_c \cdot \mu_c)
]

즉 BN은 **채널별로 `u`에 대한 선형 변환**입니다.

### 2) Conv와 결합: `W'`와 `b'`로 재표현 가능

Conv는 (개념적으로)

[
u = Conv(x; W, b) = W * x + b
]

BN이 `u`에 대해 `a_c`로 스케일되고 상수항이 더해지므로:

[
BN(u) = a \odot (W*x + b) + (\beta - a \odot \mu)
]

여기서 `a`는 채널별 스케일 벡터이고 `⊙`는 채널 축에 대한 elementwise 곱(브로드캐스트)입니다. 이를 정리하면:

* **새 weight**
  [
  W' = a \odot W
  ]
  (채널 `c`의 필터 전체에 `a_c`를 곱하는 것)

* **새 bias**
  [
  b' = a \odot (b - \mu) + \beta
  ]

결국:

[
BN(Conv(x;W,b)) \equiv Conv(x;W',b')
]

이렇게 **`W' + b'` 꼴의 Conv 하나로 치환 가능**합니다.

### 3) 적용 조건(의미 보존 관점)

* BN이 **eval/inference 모드**여야 함(= running stats가 고정 상수로 사용됨)
* `gamma/beta`가 존재하는 affine BN(또는 affine 없는 BN은 별도 처리 필요)
* 채널 수 정합(Conv의 `Cout`과 BN의 채널 수가 같아야 함)

---

## conv_add_folding.py (Conv–Add Folding)

이 pass는 다음 패턴을 대상으로 합니다.

```
y = Conv(x; W, b) + z
```

여기서 `z`가 **상수이거나 (방송 가능한) bias 형태**라면, Add를 Conv의 bias로 흡수할 수 있습니다.

### 1) 수식으로 보는 원리

Conv 결과에 Add가 붙으면:

[
y = (W*x + b) + z
]

`z`가 출력 채널에 broadcast 가능한 상수(예: `[Cout]`, 또는 `[1, Cout, 1, 1]`)라면:

[
y = W*x + (b + z)
]

따라서:

* **새 bias**
  [
  b' = b + z
  ]

로 두면:

[
Conv(x;W,b) + z \equiv Conv(x;W,b')
]

즉 Add 노드를 제거하고 Conv의 bias만 수정하면 됩니다.

### 2) 적용 조건(의미 보존 관점)

* `z`가 **런타임 입력이 아니라 상수**거나, 최소한 컴파일 타임에 결정 가능한 값이어야 함
* `z`의 shape이 Conv 출력에 **broadcast 가능**해야 함(채널별 bias 형태가 대표적)
* Conv bias가 없으면 새 bias를 생성해서 흡수 가능