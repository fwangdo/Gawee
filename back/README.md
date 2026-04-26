# Back

`back/` is the AOT execution layer for Gawee after lowering is already done.

이 디렉토리는 lowering을 하지 않는다. 대신 이미 만들어진:

- LLVM dialect MLIR (`.mlir`)
- LLVM IR (`.ll`)

을 받아서 실제로 돌려볼 수 있는 실행 경로를 만든다.

핵심 목표는 두 가지다:

1. end-to-end correctness 확인
2. end-to-end latency 측정

---

## 들어 있는 도구

- `gawee-aot`
  - lowered MLIR/LLVM IR과 ABI source MLIR을 받아 실행용 runner executable을 생성한다.
- `gawee-eval`
  - 생성된 runner를 반복 실행해 latency를 재고,
  - `.npy` 출력 디렉토리를 reference와 비교한다.
- `runtime_support.h`
  - static memref descriptor, `.npy` load/save, tensor compare 유틸리티

---

## 왜 ABI source가 필요한가?

LLVM IR만 보면 함수 호출은 가능하지만,

- 각 argument의 shape
- dtype
- rank
- 어느 인자가 input이고 어느 인자가 output buffer인지

를 사람 친화적으로 다루기 어렵다.

그래서 `gawee-aot`는 **ABI source MLIR**을 같이 받는다.

예를 들어:

```mlir
func.func @forward(%arg0: memref<1x3x224x224xf32>,
                   %arg1: memref<1x64x112x112xf32>) {
  ...
}
```

이 시그니처를 읽어서:

- `arg0.npy`를 입력으로 로드하고
- `arg1`은 zero-initialized output buffer로 만들고
- direct static memref LLVM ABI 형태로 호출한다.

즉 이 방식은 JSON manifest 없이도 돌아간다.

---

## 현재 ABI 가정

현재 runner는 MLIR의 **direct static memref LLVM ABI**를 가정한다.

예:

```mlir
memref<1x3x224x224xf32>
```

는 LLVM 함수 인자에서 대략 이렇게 평탄화된다:

```text
allocated_ptr,
aligned_ptr,
offset,
size0, size1, size2, size3,
stride0, stride1, stride2, stride3
```

`gawee-aot`는 ABI source MLIR을 읽어서 이 flattened 호출 코드를 자동 생성한다.

---

## 빌드

```bash
cmake -S back -B back/build
cmake --build back/build
```

생성물:

- `back/build/gawee-aot`
- `back/build/gawee-eval`

---

## 실행 흐름

전체 흐름은 다음과 같다.

1. middle-end가 lowered MLIR 또는 LLVM IR 생성
2. `gawee-aot build`로 실행용 runner 생성
3. runner가 `arg0.npy`, `arg1.npy` 같은 입력을 읽고 실제 실행
4. 결과를 `output0.npy`, `output1.npy`로 저장
5. `gawee-eval compare`로 reference와 correctness 비교
6. `gawee-eval benchmark`로 end-to-end latency 측정

---

## 1. Runner 생성

LLVM dialect MLIR에서 바로 시작하는 예:

```bash
back/build/gawee-aot build \
  --abi-source middle/mlir/test/llvm_test.mlir \
  --input middle/mlir/test/llvm_test.mlir \
  --output back/build/simple_loop_runner \
  --entry simple_loop \
  --num-output-args 1
```

옵션 의미:

- `--abi-source`
  - 함수 시그니처를 읽을 MLIR 파일
- `--input`
  - lowered `.mlir` 또는 `.ll`
- `--output`
  - 생성할 실행 파일 경로
- `--entry`
  - 호출할 함수 이름
- `--num-output-args`
  - 마지막 몇 개의 함수 인자를 output buffer로 볼지

여기서는:

- 함수 argument가 두 개라면
- 앞쪽은 input
- 마지막 한 개는 output

으로 해석한다.

---

## 2. Runner 실행

입력 디렉토리 구조:

```text
inputs/
  arg0.npy
  arg1.npy
  ...
```

실행:

```bash
back/build/simple_loop_runner inputs outputs
```

출력 디렉토리:

```text
outputs/
  output0.npy
  output1.npy
  ...
```

규칙:

- input 파일명은 `argN.npy`
- output 파일명은 `outputN.npy`

이다.

---

## 3. Correctness 비교

reference 출력이 `expected/`에 있다고 가정하면:

```bash
back/build/gawee-eval compare \
  --actual outputs \
  --expected expected \
  --atol 1e-5 \
  --rtol 1e-5
```

동작:

- `expected/`의 `.npy` 파일 목록을 기준으로 비교
- dtype과 shape가 먼저 같은지 확인
- `f32`, `f64`는 `atol/rtol` 기준으로 `allclose`
- `i32`, `i64`는 exact compare

출력 예:

```text
output0.npy: PASS, max_abs_diff=0
all outputs matched, max_abs_diff=0
```

---

## 4. Latency 측정

```bash
back/build/gawee-eval benchmark \
  --runner back/build/simple_loop_runner \
  --inputs inputs \
  --outputs bench_outputs \
  --warmup 3 \
  --iters 20
```

출력 예:

```text
benchmark summary (ms)
  warmup: 3
  iterations: 20
  min: 0.412
  p50: 0.425
  avg: 0.431
  p95: 0.448
  max: 0.455
```

주의:

이 수치는 **runner 전체 실행 시간**이다.
즉 아래가 모두 포함된다:

- `.npy` 입력 로드
- output buffer 준비
- compiled function 호출
- `.npy` 출력 저장

그래서 이건 kernel-only latency가 아니라,
현재 Gawee backend의 **end-to-end host-side 실행 latency**다.

---

## correctness와 latency를 같이 보려면

권장 루틴은 이렇다.

1. runner 생성
2. 한 번 실행해서 `outputs/` 생성
3. `gawee-eval compare`로 correctness 확인
4. correctness가 맞으면 `gawee-eval benchmark`로 latency 측정

즉:

- correctness가 먼저
- latency는 그 다음

순서로 보는 것이 맞다.

---

## 현재 한계

현재 범위는 intentionally narrow 하다.

- static ranked memref만 지원
- direct LLVM ABI만 지원
- `_mlir_ciface_*` wrapper는 아직 미지원
- dynamic shape 미지원
- multiple return value 미지원
- latency는 process launch + file I/O 포함

하지만 지금 단계의 목적은 충분히 달성한다:

- lowering 결과를 실제로 끝까지 실행해볼 수 있음
- output correctness를 reference와 비교할 수 있음
- 성능 작업 시작 전에 end-to-end latency baseline을 만들 수 있음

---

## 다음 확장 포인트

성능 작업으로 넘어가면 다음이 자연스럽다.

1. in-process benchmark 모드 추가
2. file I/O를 제외한 pure invocation latency 측정
3. reference backend 자동 실행 연결
4. generated runner 없이 shared library 직접 호출
5. dynamic memref support
