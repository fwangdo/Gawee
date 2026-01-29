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

---

## conv_add_folding.py (Conv–Add Folding)

이 pass는 다음 패턴을 대상으로 합니다.

```
y = Conv(x; W, b) + z
```

여기서 `z`가 **각 채널에 대한 bias 형태**라면, Add를 Conv의 bias로 흡수합니다.