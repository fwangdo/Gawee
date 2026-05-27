# Phase 10: Conv Consumer Chain Fusion Summary

## 이번에 변경한 내용

### 1. Consumer Chain 탐색 헬퍼 추가 (`findElementwiseConsumerChain`)

**개념**: conv op의 결과를 따라 forward로 걸어가면서, single-use로 연결된 elementwise `linalg.generic` op들을 수집한다. 이 chain이 바로 fuse 대상이다.

**chain 발견 조건**:
1. 현재 op의 result가 정확히 1개
2. 그 result의 user가 정확히 1개 (single-use)
3. 그 user가 `linalg.GenericOp`이고 elementwise (모든 loop이 parallel + 모든 indexing map이 projected permutation)

**예시 chain**:
```
conv (result: tensor<1x16x8x8xf32>)
  └─ single use → bias_add (linalg.generic, elementwise)
                    └─ single use → relu (linalg.generic, elementwise)
```
이 경우 `findElementwiseConsumerChain(conv)` → `[bias_add, relu]`

**왜 single-use인지**: multi-use면 중간 결과가 loop 밖에서도 필요하므로 tile loop 안에서만 계산하면 다른 user가 값을 못 받음.

### 2. Tiling Root 변경: conv → last consumer

**기존 접근**:
```
tiling root = conv
fusionControlFn: fill만 fuse (destination operand만)
결과: scf.for { fill(fused), conv(tiled) } → bias_add → relu
```

**새 접근**:
```
tiling root = relu (chain의 마지막)
fusionControlFn: fill, conv, elementwise generic 모두 fuse 허용
결과: scf.for { fill(fused), conv(fused), bias_add(fused), relu(tiled) }
```

**핵심 원리**: `tileConsumerAndFuseProducersUsingSCF`는 tiling root를 tile하고, 그 root의 input/output producer를 **역방향**으로 탐색해서 fuse한다. 따라서 chain의 마지막 op을 root로 잡으면 API가 자동으로 bias_add → conv → fill을 역방향 탐색하여 모두 fuse한다.

### 3. Tile Sizes 결정

**chain이 있을 때**:
- tiling root는 elementwise generic (4D: N,C,H,W)
- tile sizes = conv plan의 `parallelTileSizes` (e.g. `[1, 8, 8, 8]`)
- conv의 7 loops (4 parallel + 3 reduction)와 달리, elementwise는 4 loops만 있으므로 reduction dim padding 불필요

**chain이 없을 때** (fallback):
- 기존 동작 유지: conv를 root로, 7 loops에 대해 tile sizes를 pad

### 4. fusionControlFn 확장

기존: destination operand의 `linalg.fill`만 fuse
변경: `isDestinationOperand` 제한 제거, 다음 세 종류 모두 fuse 허용:

| Producer 종류 | 예시 | 왜 fuse하는가 |
|---|---|---|
| `linalg::FillOp` | conv의 zero-init | destination init, 항상 안전 |
| `linalg::Conv2DNchwFchwOp` | conv 자체 | bias_add의 input producer |
| elementwise `linalg::GenericOp` | bias_add | relu의 input producer |

**중요**: `isDestinationOperand` 체크를 제거한 이유는, conv가 bias_add의 **input** (ins) producer이지 destination이 아니기 때문. destination만 허용하면 conv가 fuse되지 않음.

### 5. Cleanup 확장

기존: consumer(conv)와 fused producers(fill)만 정리
변경: tiling root(relu), 원래 conv, chain의 모든 intermediate op(bias_add), fused producers(fill) 모두 정리

**순서 주의**: tiling root를 먼저 지우고, 그 다음 conv, chain ops, fused producers 순서. `use_empty()` 체크로 아직 사용 중인 op은 건드리지 않음.

## 왜 이게 성능에 도움이 되는가

**변경 전**:
```
scf.for {
  fill_tile → conv_tile  // tile 크기의 output 생성
}
// conv의 full-size output이 메모리에 materialize
bias_add(full_output)    // full-size tensor read + write
relu(full_output)        // full-size tensor read + write
```

**변경 후**:
```
scf.for {
  fill_tile → conv_tile → bias_add_tile → relu_tile
  // tile 크기의 intermediate만 사용, full-size 중간 텐서 없음
}
```

tile 크기가 작으므로 중간 결과가 register/L1 cache에 머물 수 있고, full-size tensor의 메모리 할당과 read/write가 사라짐.

## 추가 변경: while-loop 리팩토링

기존에는 모든 conv plan을 미리 수집한 후 순서대로 처리했음. 하지만 첫 번째 conv의 tile-and-fuse가 IR을 변경하면, 미리 수집한 두 번째 conv의 `plan.operation` 포인터가 dangling이 될 수 있음 → segfault.

**수정**: while loop으로 변경하여 한 번에 하나의 conv만 찾아서 처리하고, 처리 후 다시 walk. `gawee.transform.tiled` attribute으로 이미 처리된 conv를 skip.

## 추가 변경: residual add 감지

ResNet의 skip connection에서 오는 `add(conv_result, skip)` 같은 multi-input op이 chain에 포함되면, tiling API가 skip input의 producer까지 fuse하려 해서 잘못된 IR이 생김.

**수정**: `findElementwiseConsumerChain`에서 consumer의 모든 tensor input을 검사 — chain predecessor나 scalar/1D broadcast가 아닌 full-rank tensor input이 있으면 chain을 거기서 끊음.

## 현재 상태: chain fusion 비활성화

consumer chain fusion은 단일 conv chain 패턴에서는 정확하게 동작하지만 (tile_fuse_test.mlir, stride=2 conv, padded conv, two-conv chain 모두 검증), **resnet18 (20 convs) 전체에서 런타임 segfault가 발생**. root cause 미확인, chain 비활성화로 우회 중.

infrastructure (findElementwiseConsumerChain, fusionControlFn 확장, while-loop, tiled marker)는 모두 완성되어 있으므로, 활성화 시 resnet18 디버깅만 필요.

## gawee-opt.cpp 변경

`--gawee-linalg-transform` 플래그로 LinalgTransform pass를 단독 실행할 수 있도록 standalone pipeline 등록 추가. 이전에는 full pipeline(`--gawee-to-loops` 등)을 통해서만 실행 가능했음.
