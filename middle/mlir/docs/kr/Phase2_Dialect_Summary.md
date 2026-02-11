# Phase 2: TableGen과 Dialect 정의

## 개요

이 Phase에서는 MLIR의 **TableGen** 시스템을 사용하여 커스텀 Dialect(gawee)와 Operation들을 정의한다.
직접 C++ 클래스를 하나하나 작성하는 대신, `.td` 파일에 선언적으로 기술하면
`mlir-tblgen` 도구가 C++ 코드를 자동 생성한다.

---

## 1. TableGen이란?

### 왜 TableGen을 쓰는가?

MLIR에서 Op 하나를 정의하려면 다음이 필요하다:
- C++ 클래스 선언 (Op 이름, 인자, 결과)
- Parser / Printer (텍스트 IR ↔ 내부 표현 변환)
- Verifier (잘못된 IR 감지)
- Accessor 메서드 (getInput(), getWeight() 등)

이걸 직접 쓰면 Op 하나당 수백 줄이다. TableGen은 이 보일러플레이트를 `.td` 파일 몇 줄로 해결한다.

### 흐름

```
GaweeDialect.td  ──┐
                    ├──→ mlir-tblgen ──→ .h.inc / .cpp.inc (C++ 코드 조각)
GaweeOps.td     ──┘
```

`.inc` 파일은 **독립적으로 컴파일되지 않는다.** `#include`로 `.cpp`/`.h` 안에 삽입된다.

---

## 2. Dialect 정의: GaweeDialect.td

Dialect은 Op, Type, Attribute의 **네임스페이스**다.
`arith` dialect에 `arith.addi`가 있듯이, `gawee` dialect에 `gawee.conv`가 있다.

```tablegen
// GaweeDialect.td
include "mlir/IR/OpBase.td"   // MLIR 기본 정의 (Dialect, Op 클래스 등)

def Gawee_Dialect : Dialect {
  let name = "gawee";                          // IR에서 보이는 접두사
  let summary = "Gawee dialect for neural network graphs";
  let cppNamespace = "::mlir::gawee";          // 생성된 C++의 namespace
  let usePropertiesForAttributes = 0;          // bytecode 지원 비활성화
}
```

### 핵심 필드 설명

| 필드 | 역할 |
|------|------|
| `name` | IR 텍스트에서 사용: `gawee.conv`, `gawee.relu` 등 |
| `cppNamespace` | 생성된 C++ 코드의 네임스페이스 |
| `usePropertiesForAttributes` | 0으로 하면 레거시 속성 방식 사용 (헤더 누락 문제 회피) |

### Base Op 클래스

모든 Gawee op이 상속하는 기본 클래스:

```tablegen
class Gawee_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gawee_Dialect, mnemonic, traits>;
```

- `mnemonic`: op 이름 ("conv", "relu" 등)
- `traits`: op의 특성 (예: `Pure` = 부수효과 없음)
- `Op<Gawee_Dialect, ...>`: 이 op이 Gawee dialect에 속함을 선언

---

## 3. Operation 정의: GaweeOps.td

Op 정의는 세 부분이다: **arguments**, **results**, **summary**.

### Conv Op 예시

```tablegen
def Gawee_ConvOp : Gawee_Op<"conv", []> {
  let summary = "2D convolution";
  let arguments = (ins
    AnyTensor:$input,                 // 텐서 타입, $input은 accessor 이름
    AnyTensor:$weight,
    AnyTensor:$bias,
    DenseI64ArrayAttr:$strides,       // 정수 배열 속성
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs AnyTensor:$output);
}
```

### 속성(Attribute) 타입 정리

| 타입 | 의미 | 예시 |
|------|------|------|
| `AnyTensor` | 임의 텐서 (rank, element type 무관) | input, weight |
| `DenseI64ArrayAttr` | `int64_t` 배열 속성 | strides=[1,1], padding=[3,3] |
| `I64Attr` | 단일 `int64_t` 속성 | startDim=1 |
| `F64Attr` | 단일 `double` 속성 | eps=1e-5 |
| `BoolAttr` | 불리언 속성 | ceilMode=false |

### `$이름`의 의미

`$input`이라고 쓰면 TableGen이 자동으로 다음을 생성한다:
- `getInput()` — Value를 반환
- `getInputAttr()` — Attribute 객체를 반환 (속성 타입만)
- Op 생성 시 인자 순서 결정

### 더 많은 Op 정의

```tablegen
// 단순 elementwise op: 입력 하나, 출력 하나
def Gawee_ReluOp : Gawee_Op<"relu", []> {
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

// 두 입력을 받는 op
def Gawee_AddOp : Gawee_Op<"add", []> {
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
}

// 여러 타입의 속성을 섞은 op
def Gawee_MaxPoolOp : Gawee_Op<"max_pool", []> {
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernelSize,    // 배열 속성
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation,
    BoolAttr:$ceilMode                // 불리언 속성
  );
  let results = (outs AnyTensor:$output);
}

// I64Attr (단일 정수) 속성 사용
def Gawee_FlattenOp : Gawee_Op<"flatten", []> {
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$startDim,                // 단일 정수 속성
    I64Attr:$endDim
  );
  let results = (outs AnyTensor:$output);
}
```

---

## 4. 빌드 시스템: mlir-tblgen 호출

`build.sh`에서 TableGen을 실행하는 네 가지 커맨드:

```bash
TBLGEN="$LLVM_DIR/bin/mlir-tblgen"

# 1. Dialect 선언 생성 (.h용)
$TBLGEN --gen-dialect-decls \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.h.inc

# 2. Dialect 정의 생성 (.cpp용)
$TBLGEN --gen-dialect-defs \
  -I $LLVM_DIR/include \
  include/Gawee/GaweeDialect.td \
  -o include/Gawee/generated/GaweeDialect.cpp.inc

# 3. Op 선언 생성 (.h용)
$TBLGEN --gen-op-decls \
  -I $LLVM_DIR/include -I include -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.h.inc

# 4. Op 정의 생성 (.cpp용)
$TBLGEN --gen-op-defs \
  -I $LLVM_DIR/include -I include -I include/Gawee \
  include/Gawee/GaweeOps.td \
  -o include/Gawee/generated/GaweeOps.cpp.inc
```

### tblgen 플래그 정리

| 플래그 | 출력 | 용도 |
|--------|------|------|
| `--gen-dialect-decls` | `GaweeDialect.h.inc` | Dialect 클래스 선언 |
| `--gen-dialect-defs` | `GaweeDialect.cpp.inc` | `initialize()` 등 구현 |
| `--gen-op-decls` | `GaweeOps.h.inc` | `ConvOp`, `ReluOp` 등 선언 |
| `--gen-op-defs` | `GaweeOps.cpp.inc` | Parser, printer, accessor 구현 |

### `-I` 플래그

`include "mlir/IR/OpBase.td"` 같은 구문을 해석하기 위한 검색 경로.
- `-I $LLVM_DIR/include`: MLIR 기본 .td 파일 위치
- `-I include/Gawee`: `GaweeDialect.td`를 `GaweeOps.td`에서 찾기 위함

---

## 5. 생성된 .inc 파일의 사용

`.inc` 파일은 직접 컴파일하지 않고, `.h`/`.cpp` 안에서 `#include`한다:

```cpp
// GaweeDialect.h — 사용자가 직접 작성
namespace mlir::gawee {
#include "Gawee/generated/GaweeDialect.h.inc"  // Dialect 클래스 선언
#include "Gawee/generated/GaweeOps.h.inc"      // Op 클래스 선언
}
```

```cpp
// GaweeDialect.cpp — 사용자가 직접 작성
#include "Gawee/GaweeDialect.h"

#include "Gawee/generated/GaweeDialect.cpp.inc"  // Dialect 구현
#define GET_OP_CLASSES
#include "Gawee/generated/GaweeOps.cpp.inc"      // Op 구현
```

`#define GET_OP_CLASSES`는 생성된 `.cpp.inc` 내부의 조건부 컴파일 매크로다.
이걸 정의해야 Op 클래스 구현 코드가 활성화된다.

---

## 6. 새 Op 추가 체크리스트

새 연산을 추가할 때의 최소 작업:

1. `GaweeOps.td`에 `def Gawee_XxxOp` 추가
2. `build.sh` 재실행 (tblgen이 .inc 재생성)
3. `GaweeToLinalg.cpp`에 lowering 패턴 추가
4. `MLIREmitter.cpp`에 emit 함수 추가
5. `gawee-opt.cpp`의 `patterns.add<>` 에 등록

---

## 핵심 개념 정리

- **TableGen** = 선언적 Op 정의 언어. `.td` → `.inc` (C++ 코드 조각)
- **Dialect** = Op들의 네임스페이스. `gawee.conv`에서 `gawee`가 dialect
- **Op 정의** = `arguments` (입력) + `results` (출력) + `summary`
- **속성 타입** = `AnyTensor`, `DenseI64ArrayAttr`, `I64Attr`, `BoolAttr`, `F64Attr`
- **$이름** = TableGen이 자동으로 getter 메서드를 생성하는 키
- **.inc 파일** = 독립 컴파일 불가, `#include`로 삽입해야 함
