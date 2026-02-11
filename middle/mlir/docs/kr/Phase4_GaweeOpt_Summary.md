# Phase 4: gawee-opt 도구

## 개요

`gawee-opt`는 MLIR IR을 읽어서 Pass(변환)를 적용하는 도구다.
MLIR의 `MlirOptMain` 함수를 사용하여 커맨드라인 파싱, 파일 읽기, Pass 실행을 자동 처리한다.

```
gawee-opt --convert-gawee-to-linalg input.mlir   # Gawee → Linalg
gawee-opt --gawee-to-llvm input.mlir              # 전체 파이프라인
```

---

## 1. main() 함수의 세 가지 역할

```cpp
int main(int argc, char **argv) {
  // (1) Pass 등록
  // (2) Dialect 등록
  // (3) MlirOptMain 호출
}
```

이 세 단계를 이해하면 gawee-opt 전체를 이해한 것이다.

---

## 2. Pass 등록: PassPipelineRegistration

```cpp
PassPipelineRegistration<>(
    "convert-gawee-to-linalg",                    // CLI 플래그 이름
    "Lower Gawee dialect to Linalg dialect",      // 설명
    [](OpPassManager &pm) {
      pm.addPass(gawee::createGaweeToLinalgPass());
    });
```

이렇게 하면 `--convert-gawee-to-linalg` 옵션이 CLI에 등록된다.

### 복합 파이프라인

여러 Pass를 묶어 하나의 파이프라인으로 등록할 수 있다:

```cpp
PassPipelineRegistration<>(
    "gawee-to-llvm",
    "Full pipeline: Gawee -> Linalg -> SCF -> LLVM dialect",
    [](OpPassManager &pm) {
      // Step 1: Gawee → Linalg
      pm.addPass(gawee::createGaweeToLinalgPass());
      // Step 2: tensor.empty → bufferization.alloc_tensor
      pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
      // Step 3: Bufferize (tensor → memref)
      bufferization::OneShotBufferizePassOptions bufOpts;
      bufOpts.bufferizeFunctionBoundaries = true;
      pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
      // Step 4: Linalg → SCF loops
      pm.addPass(createConvertLinalgToLoopsPass());
      // Step 5: SCF → ControlFlow
      pm.addPass(createSCFToControlFlowPass());
      // Step 6: → LLVM dialect
      pm.addPass(createArithToLLVMConversionPass());
      pm.addPass(createConvertControlFlowToLLVMPass());
      pm.addPass(createFinalizeMemRefToLLVMConversionPass());
      pm.addPass(createConvertFuncToLLVMPass());
      // Step 7: unrealized_cast 정리
      pm.addPass(createReconcileUnrealizedCastsPass());
    });
```

---

## 3. Dialect 등록: DialectRegistry

```cpp
DialectRegistry registry;

// 우리 dialect
registry.insert<gawee::GaweeDialect>();

// 변환 결과로 생성되는 dialect들
registry.insert<linalg::LinalgDialect>();
registry.insert<arith::ArithDialect>();
registry.insert<tensor::TensorDialect>();
registry.insert<func::FuncDialect>();
registry.insert<scf::SCFDialect>();
registry.insert<memref::MemRefDialect>();
registry.insert<bufferization::BufferizationDialect>();
registry.insert<cf::ControlFlowDialect>();
registry.insert<LLVM::LLVMDialect>();
```

### 왜 모든 dialect을 등록해야 하는가?

MLIR은 dialect을 **지연 로딩(lazy loading)**한다. 등록하지 않은 dialect의 op을 만나면 파싱 오류가 발생한다.
파이프라인 중간 단계에서 생성되는 모든 dialect을 미리 등록해야 한다.

---

## 4. Bufferization 인터페이스 등록

```cpp
arith::registerBufferizableOpInterfaceExternalModels(registry);
linalg::registerBufferizableOpInterfaceExternalModels(registry);
tensor::registerBufferizableOpInterfaceExternalModels(registry);
bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
```

One-Shot Bufferize는 각 dialect의 op을 어떻게 bufferize할지 알아야 한다.
이 함수들이 그 정보("이 op은 이렇게 memref로 변환해라")를 등록한다.

등록하지 않으면 bufferize 시 "don't know how to bufferize this op" 오류가 발생한다.

---

## 5. MlirOptMain

```cpp
return mlir::asMainReturnCode(
    mlir::MlirOptMain(argc, argv, "Gawee MLIR Optimizer\n", registry));
```

`MlirOptMain`이 처리하는 것:
- `argc, argv` 파싱 (입력 파일, 출력 파일, pass 옵션)
- MLIR 텍스트 파싱
- 등록된 pass 실행
- 결과 출력

**우리는 main()에서 "무엇을 등록할지"만 결정하고, 나머지는 MLIR 프레임워크에 위임한다.**

---

## 6. RTTI 문제와 -fno-rtti

CMakeLists.txt에 다음이 있다:

```cmake
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()
```

LLVM은 빌드 시 RTTI(Runtime Type Information)를 비활성화한다.
우리 코드가 RTTI를 사용하면 링크 오류가 발생하므로 동일하게 비활성화해야 한다.

**RTTI란?** `dynamic_cast`, `typeid` 같은 C++ 런타임 타입 검사 기능.
LLVM은 성능상의 이유로 이를 끄고 대신 자체 RTTI(`isa<>`, `cast<>`, `dyn_cast<>`)를 사용한다.

---

## 7. getDependentDialects의 역할

Pass 정의에서:
```cpp
void getDependentDialects(DialectRegistry &registry) const override {
  registry.insert<linalg::LinalgDialect>();
  registry.insert<arith::ArithDialect>();
  registry.insert<tensor::TensorDialect>();
}
```

이것은 "이 pass를 실행하면 linalg, arith, tensor op이 생성됩니다"라고 프레임워크에 알려주는 것이다.
Pass 매니저가 실행 전에 필요한 dialect을 미리 로드할 수 있도록 한다.

---

## 8. CMakeLists.txt 구조

```cmake
# gawee-opt 바이너리 정의
add_executable(gawee-opt tools/gawee-opt.cpp)

# 링크할 라이브러리 — 순서 중요!
target_link_libraries(gawee-opt
  PRIVATE
  GaweeDialect          # 우리 dialect
  GaweeConversion       # 우리 lowering pass
  MLIROptLib            # MlirOptMain 함수
  MLIRParser            # MLIR 텍스트 파싱
  MLIRPass              # Pass 인프라
  MLIRLinalgDialect     # Linalg ops
  MLIRBufferizationTransforms  # Bufferize pass
  MLIRLLVMDialect       # LLVM dialect
  # ... 기타 dialect과 변환
)
```

**link 순서가 중요한 이유**: 정적 라이브러리에서 심볼 해결은 순서에 의존한다.
의존하는 라이브러리가 먼저 오고, 의존받는 라이브러리가 나중에 와야 한다.

---

## 핵심 개념 정리

- **PassPipelineRegistration** = CLI 옵션으로 pass 파이프라인 등록
- **DialectRegistry** = 사용할 모든 dialect을 등록 (지연 로딩)
- **Bufferization 인터페이스** = 각 dialect의 bufferize 방법 등록
- **MlirOptMain** = MLIR 프레임워크의 표준 opt 진입점
- **-fno-rtti** = LLVM과 동일한 RTTI 설정 맞추기
- **getDependentDialects** = pass가 생성하는 dialect 선언
