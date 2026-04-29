# MatMul Lowering Summary

이번 변경에서 핵심은 `ONNX MatMul`과 `Gemm`를 같은 계약으로 보지 않는 것이다.

- `Gemm`
  보통 `Y = A * B + C` 형태의 linear/projection 의미가 강하다.
  현재 코드베이스에서는 `gawee.linear` 경로가 이 역할에 가깝다.

- `MatMul`
  ONNX의 일반 행렬곱이다.
  attention처럼 batch 차원이 여러 개 붙은 `[..., M, K] x [..., K, N] -> [..., M, N]`도 포함한다.

## 왜 분리해야 하나

`MatMul`을 `linear`로 보내면 아래 가정이 숨어 들어간다.

- weight가 사실상 2D다
- 마지막 축 projection이다
- bias를 붙일 수 있거나 붙여도 의미가 같다
- batch broadcast 규칙을 별도로 고려하지 않아도 된다

이 가정은 attention score/value matmul에는 맞지 않는다.

## 이번 lowering의 구조

`gawee.matmul`은 `linalg.generic`로 낮춘다.

- 출력 loop: `batch dims + M + N`
- reduction loop: `K`
- iterator type:
  - batch dims, `M`, `N`는 `parallel`
  - `K`는 `reduction`

개념적으로는 다음과 같다.

```text
for batch...
  for m
    for n
      acc = 0
      for k
        acc += lhs[..., m, k] * rhs[..., k, n]
```

## indexing map에서 이해할 점

출력 loop 차원은 `[batch..., m, n, k]` 순서로 잡는다.

- lhs map: `batch..., m, k`
- rhs map: `batch..., k, n`
- out map: `batch..., m, n`

즉 reduction 축 `k`는 출력에는 나타나지 않고, 입력 두 개만 공유한다.

## batch broadcast

ONNX MatMul은 batch prefix에 broadcast가 있다.
이번 구현은 다음 방식으로 처리한다.

- operand batch dim이 정적 `1`이면 affine map에서 `0` 인덱스로 고정
- 그렇지 않으면 해당 loop batch dim을 그대로 사용

즉 `shape = 1`인 batch dim은 같은 값을 반복 재사용한다.

## 직접 다시 구현하려면 알아야 할 것

1. `Gemm`과 `MatMul`의 의미 차이를 설명할 수 있어야 한다.
2. `linalg.generic`의 input/output indexing map을 손으로 써볼 수 있어야 한다.
3. reduction 축이 output indexing map에 왜 없는지 이해해야 한다.
4. dynamic output shape를 만들 때 어느 operand의 `dim`을 읽어야 하는지 설명할 수 있어야 한다.

