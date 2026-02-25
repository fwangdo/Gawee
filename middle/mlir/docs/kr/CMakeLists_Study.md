# CMakeLists.txt 완전 해부

이 문서는 `middle/mlir/CMakeLists.txt`의 모든 줄을 초보자 관점에서 설명합니다.

---

## 전체 구조 요약

이 CMakeLists.txt는 크게 5단계로 구성됩니다:

```
1. 프로젝트 기본 설정 (C++ 버전, 컴파일 옵션)
2. MLIR/LLVM 찾기 & CMake 모듈 로드
3. 라이브러리 빌드 (GaweeDialect, GaweeConversion, GaweeEmit)
4. 실행 파일 빌드 (gawee-opt, gawee-translate)
5. 링크 (내가 만든 코드와 LLVM/MLIR 라이브러리 연결)
```

---

## 1단계: 프로젝트 기본 설정 (1~6줄)

```cmake
cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR LANGUAGES C CXX)
```

- `cmake_minimum_required`: "CMake 3.20 이상이어야 빌드 가능" 선언
- `project(GaweeMLIR ...)`: 프로젝트 이름을 `GaweeMLIR`로 정하고, C와 C++ 컴파일러를 사용한다고 선언

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

- C++17 표준을 사용하겠다는 선언
- `REQUIRED ON`: C++17을 지원하지 않는 컴파일러면 빌드 자체를 거부

```cmake
add_compile_options(-Wno-reserved-user-defined-literal)
```

- LLVM/MLIR 헤더에서 발생하는 특정 경고를 끄는 옵션
- 우리 코드 문제가 아니라 LLVM 헤더 내부의 문법 때문에 나오는 경고를 무시

```cmake
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
```

- **RTTI(Run-Time Type Information)**: C++에서 `dynamic_cast`나 `typeid` 같은 기능을 가능하게 하는 런타임 타입 정보
- LLVM은 성능상의 이유로 RTTI를 **끄고** 빌드됨
- 우리 코드도 LLVM과 동일한 설정을 맞춰야 함. 안 그러면 링크 에러 발생
- **핵심**: LLVM과 링크되는 코드는 LLVM의 빌드 옵션을 그대로 따라야 한다

---

## 2단계: MLIR/LLVM 찾기 (18~27줄) — 가장 중요한 부분

```cmake
find_package(MLIR REQUIRED CONFIG)
```

### `find_package`가 하는 일

CMake에게 "내 시스템 어딘가에 MLIR이 설치되어 있으니 찾아라"라고 지시합니다.

`CONFIG` 모드에서는 `MLIRConfig.cmake`라는 파일을 찾습니다.
이 파일은 MLIR을 빌드/설치할 때 자동으로 생성된 것이며,
다음과 같은 정보를 제공합니다:

- MLIR 헤더 파일 위치 (`MLIR_INCLUDE_DIRS`)
- MLIR 라이브러리 위치
- MLIR이 사용하는 CMake 매크로/함수 위치 (`MLIR_CMAKE_DIR`)
- LLVM 관련 정보도 함께 (`LLVM_DIR`, `LLVM_INCLUDE_DIRS` 등)

실제 파일 위치 (이 시스템):
```
/Users/hdy/llvm-install/lib/cmake/mlir/MLIRConfig.cmake
```

### `find_package`가 `MLIRConfig.cmake`를 어떻게 찾는가?

CMake는 다음 순서로 찾습니다:

1. `-DMLIR_DIR=...` 옵션으로 직접 지정한 경로
2. `CMAKE_PREFIX_PATH` 환경변수에 포함된 경로
3. 시스템 기본 경로 (`/usr/local/lib/cmake/mlir/` 등)

우리는 `build.sh`에서 이렇게 지정합니다:
```bash
cmake .. -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir
```

→ 이 경로에서 `MLIRConfig.cmake`를 찾고, 거기서 모든 경로 정보를 가져옴

```cmake
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
```

- 빌드할 때 실제로 어떤 MLIR/LLVM을 사용하는지 출력 (디버깅용)

---

### CMake 모듈 로드 (23~27줄)

```cmake
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
```

- `CMAKE_MODULE_PATH`: CMake가 `include()`로 파일을 찾을 때 검색하는 경로 목록
- MLIR과 LLVM이 제공하는 CMake 유틸리티 파일들이 있는 디렉토리를 검색 경로에 추가

```cmake
include(TableGen)
include(AddLLVM)
include(AddMLIR)
```

**여기가 질문하신 `AddLLVM`입니다.**

### `include()`가 하는 일

`include(AddLLVM)`은 `CMAKE_MODULE_PATH`에 등록된 경로에서
`AddLLVM.cmake` 파일을 찾아서 **그 안의 함수/매크로를 현재 CMakeLists에 로드**합니다.

파이썬으로 비유하면:
```python
import AddLLVM  # AddLLVM.cmake 안의 함수들을 사용할 수 있게 됨
```

### `AddLLVM.cmake`는 어디에 있는가?

```
/Users/hdy/llvm-install/lib/cmake/llvm/AddLLVM.cmake
```

이 파일은 **LLVM을 빌드할 때 자동으로 생성**된 것입니다.
우리가 만든 것이 아닙니다.

### `AddLLVM.cmake`가 제공하는 것

이 파일 안에는 LLVM 프로젝트에서 사용하는 CMake 헬퍼 함수들이 정의되어 있습니다:

| 함수 | 하는 일 |
|------|---------|
| `llvm_map_components_to_libnames()` | LLVM 컴포넌트 이름 → 실제 라이브러리 이름 변환 |
| `add_llvm_library()` | LLVM 스타일로 라이브러리 추가 |
| `add_llvm_executable()` | LLVM 스타일로 실행파일 추가 |
| `add_llvm_tool()` | LLVM 도구(opt, llc 등) 추가 |

**우리 CMakeLists에서 실제로 사용하는 것:**

```cmake
llvm_map_components_to_libnames(llvm_libs Support Core)
```

이것은 `AddLLVM.cmake`에 정의된 함수입니다.
`Support`와 `Core`라는 LLVM 컴포넌트 이름을 받아서,
실제 링크해야 할 라이브러리 이름(예: `LLVMSupport`, `LLVMCore`)으로 변환하여
`llvm_libs` 변수에 저장합니다.

### 나머지 include도 같은 원리

| include | 파일 위치 | 제공하는 것 |
|---------|----------|------------|
| `include(TableGen)` | `llvm-install/lib/cmake/llvm/TableGen.cmake` | `mlir_tablegen()` 등 TableGen 관련 함수 |
| `include(AddLLVM)` | `llvm-install/lib/cmake/llvm/AddLLVM.cmake` | `llvm_map_components_to_libnames()` 등 |
| `include(AddMLIR)` | `llvm-install/lib/cmake/mlir/AddMLIR.cmake` | `add_mlir_dialect()` 등 MLIR 전용 함수 |

---

## 3단계: Include 경로 설정 (33~37줄)

```cmake
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee/generated)
```

컴파일러에게 "헤더 파일을 이 경로들에서 찾아라"라고 지시합니다.

C++에서 `#include "GaweeDialect.h"`를 쓰면,
컴파일러가 이 경로들을 순서대로 뒤져서 해당 헤더를 찾습니다.

| 경로 | 무엇이 있는가 |
|------|-------------|
| `${LLVM_INCLUDE_DIRS}` | LLVM 헤더 (`llvm/Support/...` 등) |
| `${MLIR_INCLUDE_DIRS}` | MLIR 헤더 (`mlir/IR/...`, `mlir/Dialect/...` 등) |
| `${CMAKE_SOURCE_DIR}/include` | 우리 프로젝트 헤더 (MLIREmitter.h 등) |
| `.../include/Gawee` | Dialect 헤더 (GaweeDialect.h) |
| `.../include/Gawee/generated` | TableGen이 자동 생성한 `.h.inc` 파일들 |

---

## 4단계: 라이브러리 빌드 (43~68줄)

### GaweeDialect 라이브러리

```cmake
add_library(GaweeDialect
  lib/Gawee/GaweeDialect.cpp
)

target_link_libraries(GaweeDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)
```

**`add_library(이름 소스파일)`**: 소스 파일을 컴파일하여 라이브러리(`.a` 파일)를 만든다.

- `GaweeDialect.cpp` → 컴파일 → `libGaweeDialect.a`

**`target_link_libraries(A PUBLIC B C)`**: "A는 B와 C에 의존한다"는 선언.

- `MLIRIR`: MLIR의 핵심 IR 라이브러리 (Operation, Value, Type 등)
- `MLIRSupport`: MLIR 유틸리티 (StringRef, LogicalResult 등)

**`PUBLIC`의 의미**: GaweeDialect에 링크된 다른 타겟도 자동으로 MLIRIR, MLIRSupport를 사용할 수 있음.

비유:
```
PUBLIC  = "나도 쓰고, 나를 쓰는 사람도 쓸 수 있음"
PRIVATE = "나만 쓰고, 나를 쓰는 사람은 모름"
```

### GaweeConversion 라이브러리

```cmake
add_library(GaweeConversion
  lib/Conversion/GaweeToLinalg.cpp
)

target_link_libraries(GaweeConversion
  PUBLIC
  GaweeDialect          # 우리 Dialect 정의
  MLIRLinalgDialect      # Linalg dialect (lowering 대상)
  MLIRTensorDialect      # tensor 타입
  MLIRArithDialect       # arith 연산 (상수 생성 등)
  MLIRTransforms         # 변환 프레임워크
)
```

- GaweeConversion은 GaweeDialect에 의존 → GaweeDialect가 먼저 빌드됨
- Linalg으로 lowering하므로 Linalg 관련 라이브러리도 필요

### 의존 관계 그래프

```
GaweeDialect  ←── GaweeConversion  ←── gawee-opt
     ↑                                      ↑
     └──────── GaweeEmit ←── gawee-translate │
                                             │
              MLIRIR, MLIRSupport ────────────┘
```

화살표 방향: "A ← B"는 "B가 A에 의존한다"

---

## 5단계: 실행 파일 빌드 (75~113줄)

### gawee-opt

```cmake
add_executable(gawee-opt
  tools/gawee-opt.cpp
)
```

- `add_executable`: 라이브러리가 아닌 **실행 가능한 바이너리**를 만듦
- `add_library`와 다른 점: 결과물이 `.a`가 아니라 직접 실행할 수 있는 프로그램

```cmake
llvm_map_components_to_libnames(llvm_libs Support Core)
```

- `AddLLVM.cmake`에서 가져온 함수 (위에서 설명)
- `Support` → `LLVMSupport`, `Core` → `LLVMCore`로 변환하여 `llvm_libs` 변수에 저장

```cmake
target_link_libraries(gawee-opt
  PRIVATE
  GaweeDialect
  GaweeConversion
  MLIROptLib          # mlir-opt의 메인 루프
  MLIRParser          # .mlir 파일 파싱
  MLIRPass            # pass 매니저
  ...
  ${llvm_libs}        # LLVMSupport, LLVMCore
)
```

**`PRIVATE`인 이유**: gawee-opt는 최종 실행 파일이므로, 다른 누구도 gawee-opt에 링크하지 않음. 따라서 PUBLIC일 필요가 없음.

**라이브러리가 이렇게 많은 이유**:

`gawee-opt`는 전체 파이프라인을 실행하는 도구이므로,
파이프라인의 모든 단계에서 사용하는 dialect와 변환을 전부 링크해야 합니다:

```
gawee-opt이 사용하는 것들:

Gawee → Linalg:    GaweeDialect, GaweeConversion, MLIRLinalgDialect
Linalg → SCF:      MLIRLinalgTransforms, MLIRSCFDialect
Bufferization:     MLIRBufferizationDialect, MLIRBufferizationTransforms, MLIRMemRefDialect
SCF → LLVM:        MLIRLLVMDialect, MLIRSCFToControlFlow, MLIRFuncToLLVM, ...
기반:              MLIRIR, MLIRSupport, MLIRParser, MLIRPass
```

**하나라도 빠지면 링크 에러**가 납니다. 이것이 LLVM/MLIR 프로젝트에서 빌드 에러가 자주 나는 주된 이유입니다.

---

## 전체 흐름 정리: CMake가 실행될 때 일어나는 일

```
1. cmake .. -DMLIR_DIR=~/llvm-install/lib/cmake/mlir
   │
   ├─ find_package(MLIR)
   │   └─ MLIRConfig.cmake를 읽어서 경로 변수들 설정
   │
   ├─ include(AddLLVM)
   │   └─ AddLLVM.cmake 로드 → llvm_map_components_to_libnames() 사용 가능
   │
   ├─ include_directories(...)
   │   └─ 컴파일러에게 헤더 검색 경로 전달
   │
   ├─ add_library(GaweeDialect ...)
   │   └─ GaweeDialect.cpp → libGaweeDialect.a
   │
   ├─ add_library(GaweeConversion ...)
   │   └─ GaweeToLinalg.cpp → libGaweeConversion.a
   │
   ├─ add_executable(gawee-opt ...)
   │   └─ gawee-opt.cpp + 모든 라이브러리 → gawee-opt 바이너리
   │
   └─ target_link_libraries(...)
       └─ 의존성 그래프에 따라 링크 순서 자동 결정

2. make (또는 ninja)
   │
   └─ 위에서 결정된 순서대로 컴파일 & 링크 실행
```

---

## 자주 발생하는 빌드 에러와 원인

| 에러 메시지 | 원인 | 해결 |
|------------|------|------|
| `undefined reference to mlir::XXX` | `target_link_libraries`에 해당 라이브러리 누락 | 해당 MLIR 라이브러리 추가 |
| `fatal error: 'XXX.h' file not found` | `include_directories`에 경로 누락 | 경로 추가 |
| `Could not find MLIRConfig.cmake` | `-DMLIR_DIR` 경로가 잘못됨 | 올바른 경로 지정 |
| `RTTI 관련 링크 에러` | LLVM은 RTTI 끔, 우리 코드는 켬 | `-fno-rtti` 추가 |

---

## 핵심 요약

1. **`find_package(MLIR)`**: 시스템에 설치된 MLIR을 찾아서 경로 변수들을 설정
2. **`include(AddLLVM)`**: LLVM이 제공하는 CMake 헬퍼 함수를 로드 (우리가 만든 게 아님)
3. **`add_library`**: 소스 → 라이브러리 (.a)
4. **`add_executable`**: 소스 + 라이브러리들 → 실행 파일
5. **`target_link_libraries`**: "A는 B에 의존한다" 선언. 하나라도 빠지면 링크 에러
6. **PUBLIC vs PRIVATE**: 다른 타겟에도 전파할지 여부
