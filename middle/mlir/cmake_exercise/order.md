# Study Order

이 폴더의 목적은 `한 파일 통째로 따라치기 -> 바로 build 확인`이다.

중요 원칙:
- 처음부터 전부 한 번에 쓰지 않는다.
- 한 파일만 집중해서 다시 친다.
- 파일 하나를 채운 뒤 즉시 build한다.
- build가 통과하면 다음 파일로 넘어간다.

## 0. 최초 1회 설정

```bash
cd middle/mlir/cmake_exercise
cmake -S . -B build -DMLIR_DIR=$HOME/llvm-install/lib/cmake/mlir
```

이 단계는 처음 한 번만 하면 된다.

## 1. `src/main.cpp`

가장 먼저 한다.

이 파일은 production에 정확히 1:1 대응하는 파일이 없어서
예외적으로 reference를 남겨둔다.

이 파일에서 익혀야 하는 것:
- dialect registry
- `MLIRContext`
- 작은 MLIR module parse
- `PassManager`
- pass add / run

빌드:

```bash
cmake --build build --target exercise-sandbox
```

실행:

```bash
./build/exercise-sandbox
```

## 2. `tools/exercise-opt.cpp`

그 다음 한다.

이 파일은 [middle/mlir/tools/gawee-opt.cpp](/Users/hdy/code/portfolio/Gawee/middle/mlir/tools/gawee-opt.cpp)
를 통째로 보고 다시 치는 것이 목표다.

빌드:

```bash
cmake --build build --target exercise-opt
```

실행:

```bash
./build/exercise-opt --help
```

## 3. `include/Conversion/GaweePasses.h`

그 다음 한다.

이 파일도 production header를 보고 통째로 다시 친다.

빌드:

```bash
cmake --build build --target ExerciseGaweeConversion
```

## 4. Pass 구현 순서

이제부터는 `lib/Conversion`을 파일 단위로 통째로 다시 친다.

추천 순서:
1. `GaweeToLinalg.cpp`
2. `LinalgTransform.cpp`
3. `LinalgFusion.cpp`
4. `LinalgScheduling.cpp`
5. `LinalgVectorization.cpp`
6. `LinalgVerification.cpp`
7. `BufferizePrep.cpp`
8. `DecomposeAggregatedLinalgOps.cpp`

이 순서를 추천하는 이유:
- 먼저 `Gawee -> Linalg`를 이해해야 이후 pass가 자연스럽다.
- 그 다음은 `Linalg` 단계 pass들을 차례대로 붙인다.
- 마지막에 bufferization prep과 decomposition 보조 pass를 본다.

각 파일마다 반복:

```bash
cmake --build build --target ExerciseGaweeConversion
```

필요하면 전체도 확인:

```bash
cmake --build build
```

## 5. 파일 하나를 연습하는 방법

각 파일마다 다음 순서를 고정한다.

1. 실제 production 파일을 읽는다.
   - 예: `middle/mlir/lib/Conversion/GaweeToLinalg.cpp`
2. 여기 `cmake_exercise`의 같은 이름 파일을 연다.
3. 파일 전체를 직접 다시 친다.
4. 바로 build한다.
5. 에러를 보고 고친다.
6. build가 되면 실행까지 확인한다.

## 6. build 타깃 요약

- `src/main.cpp`
  - `cmake --build build --target exercise-sandbox`
- `tools/exercise-opt.cpp`
  - `cmake --build build --target exercise-opt`
- `include/Conversion/GaweePasses.h`
  - `cmake --build build --target ExerciseGaweeConversion`
- `lib/Conversion/*.cpp`
  - `cmake --build build --target ExerciseGaweeConversion`
- 전체 확인
  - `cmake --build build`

## 7. 최종 목표

최종 목표는 단순히 코드를 읽는 것이 아니다.

목표:
- `main.cpp`를 스스로 다시 쓸 수 있다.
- `exercise-opt.cpp`를 스스로 다시 쓸 수 있다.
- 각 pass 파일을 production 코드를 참고해 복원할 수 있다.
- 최종적으로 `onnx -> gawee -> linalg -> llvm ir` 흐름에서
  `lib/Conversion`이 어떤 역할을 하는지 설명할 수 있다.

## 8. 핵심 원칙

- `main.cpp`만 reference가 있다.
- 나머지는 가능한 한 `# TODO`만 남겨 둔다.
- pass 파일들은 부분 scaffold를 보기보다
  **production 파일 전체를 다시 치는 방식**으로 연습한다.
