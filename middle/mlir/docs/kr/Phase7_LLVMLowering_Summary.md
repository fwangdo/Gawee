# Phase 7: LLVM Lowering

## 개요

Phase 5에서 Linalg → SCF 루프로 변환했다.
이 Phase에서는 SCF → ControlFlow → LLVM dialect로 최종 변환한다.
LLVM dialect까지 내려가면 `mlir-translate`로 LLVM IR로 변환 후 실행 가능한 바이너리를 만들 수 있다.

```
SCF loops → cf.br/cf.cond_br → LLVM IR ops
memref    → LLVM struct {ptr, ptr, offset, sizes, strides}
arith ops → LLVM intrinsics
```

---

## 1. 변환 단계 (gawee-to-llvm 파이프라인)

```cpp
// Step 5: SCF → ControlFlow
pm.addPass(createSCFToControlFlowPass());

// Step 6: 각 dialect → LLVM
pm.addPass(createArithToLLVMConversionPass());
pm.addPass(createConvertControlFlowToLLVMPass());
pm.addPass(createFinalizeMemRefToLLVMConversionPass());
pm.addPass(createConvertFuncToLLVMPass());

// Step 7: unrealized_cast 정리
pm.addPass(createReconcileUnrealizedCastsPass());
```

### 각 Pass의 역할

| Pass | 변환 내용 |
|------|-----------|
| `SCFToControlFlow` | `scf.for` → `cf.br` + `cf.cond_br` (기본 분기) |
| `ArithToLLVM` | `arith.addf` → `llvm.fadd` 등 |
| `ControlFlowToLLVM` | `cf.br` → `llvm.br` 등 |
| `MemRefToLLVM` | `memref.load/store/alloc` → LLVM 포인터 연산 |
| `FuncToLLVM` | `func.func` → `llvm.func`, 호출 규약 설정 |
| `ReconcileUnrealizedCasts` | 남은 `unrealized_conversion_cast` 제거 |

---

## 2. SCF → ControlFlow

SCF는 **구조화된** 제어 흐름이다. ControlFlow(cf)는 **비구조화된** 기본 분기다.

```
변환 전 (SCF):
  scf.for %i = 0 to 10 step 1 {
    // body
  }

변환 후 (ControlFlow):
  cf.br ^header(%c0)         // header 블록으로 점프
  ^header(%i):
    %cond = arith.cmpi slt, %i, %c10
    cf.cond_br %cond, ^body, ^exit   // 조건 분기
  ^body:
    // body
    %next = arith.addi %i, %c1
    cf.br ^header(%next)     // header로 돌아감
  ^exit:
```

이것이 일반적인 `for` 루프의 CFG(Control Flow Graph) 표현이다.

---

## 3. MemRef → LLVM

MemRef는 LLVM에서 **구조체(struct)**로 표현된다:

```
memref<1x64x112x112xf32>

→

llvm.struct<{
  ptr,            // allocated pointer (할당된 메모리)
  ptr,            // aligned pointer (정렬된 메모리, 실제 사용)
  i64,            // offset
  array<4 x i64>, // sizes  [1, 64, 112, 112]
  array<4 x i64>  // strides [802816, 12544, 112, 1]
}>
```

### 왜 pointer가 2개인가?

- `allocated pointer`: `malloc`이 반환한 원본 포인터 (해제 시 사용)
- `aligned pointer`: 메모리 정렬 후의 실제 데이터 시작 주소

### memref 연산의 LLVM 변환

```
memref.load %m[%i, %j]
→
// 선형 인덱스 계산: offset + i * stride[0] + j * stride[1]
%linear = llvm.add(offset, llvm.mul(%i, stride0), llvm.mul(%j, stride1))
%ptr = llvm.getelementptr %base[%linear]
%val = llvm.load %ptr
```

```
memref.alloc() : memref<1x64xf32>
→
%size = llvm.constant 64   // 1 * 64 * sizeof(f32)
%ptr = llvm.call @malloc(%size)
```

---

## 4. unrealized_conversion_cast

변환 과정에서 dialect 간 타입이 일치하지 않으면 임시로 `unrealized_conversion_cast`가 삽입된다:

```mlir
// MemRef pass는 memref를 LLVM struct로 변환했지만,
// Func pass가 아직 실행되지 않아 함수 시그니처는 memref를 기대함
%0 = unrealized_conversion_cast %llvm_struct : !llvm.struct<...> to memref<...>
```

모든 변환이 끝나면 이 cast들이 상쇄(cancellation)되어야 한다.
`reconcile-unrealized-casts` pass가 이를 확인하고 제거한다.

**만약 상쇄되지 않으면?** → 변환 누락이 있다는 뜻. 어떤 op이 LLVM으로 변환되지 못한 것이다.

---

## 5. Pass 순서가 중요한 이유

```
잘못된 순서: FuncToLLVM → MemRefToLLVM
  → func 시그니처의 memref가 먼저 LLVM으로 변환됨
  → 나중에 MemRefToLLVM이 실행될 때 이미 func이 LLVM 형태라 충돌

올바른 순서: MemRefToLLVM → FuncToLLVM
  → memref op이 먼저 변환됨
  → func은 마지막에 전체를 LLVM으로 래핑
```

실제 gawee-opt의 순서:
1. `ArithToLLVM` — 산술 연산 먼저
2. `ControlFlowToLLVM` — 분기 명령어
3. `MemRefToLLVM` — 메모리 연산
4. `FuncToLLVM` — 함수 래핑 (마지막)
5. `ReconcileUnrealizedCasts` — 정리 (반드시 마지막)

---

## 6. mlir-translate

LLVM dialect까지 내려간 후, 실제 LLVM IR로 변환:

```bash
# MLIR LLVM dialect → LLVM IR (.ll)
mlir-translate --mlir-to-llvmir output.mlir -o output.ll

# LLVM IR → 오브젝트 파일
llc output.ll -o output.o

# 링크 → 실행 파일
clang output.o -o executable
```

이 단계는 MLIR 밖의 LLVM 도구체인을 사용한다.

---

## 7. 전체 파이프라인 요약

```
JSON → gawee-translate → gawee.mlir
     → gawee-opt --gawee-to-llvm → llvm.mlir
     → mlir-translate --mlir-to-llvmir → output.ll
     → llc → output.o
     → clang → executable
```

각 단계별 IR:
```
gawee.conv          (Phase 3: 고수준)
↓
linalg.conv         (Phase 3: 중간 수준, tensor)
↓
linalg.conv (memref) (Phase 5: bufferized)
↓
scf.for + arith     (Phase 5: 루프)
↓
cf.br + arith       (Phase 7: 비구조 분기)
↓
llvm.br + llvm.fadd (Phase 7: LLVM dialect)
↓
LLVM IR (.ll)       (mlir-translate)
```

---

## 핵심 개념 정리

- **SCF → CF**: 구조화된 루프 → 기본 블록 + 분기
- **MemRef → LLVM struct**: `{alloc_ptr, aligned_ptr, offset, sizes, strides}`
- **unrealized_conversion_cast**: 중간 변환 단계의 타입 불일치를 임시 해결, 마지막에 제거
- **Pass 순서 중요**: Arith → CF → MemRef → Func → Reconcile 순서
- **mlir-translate**: LLVM dialect → LLVM IR (.ll) 변환 도구
