// ==========================================================================
// Phase 4 퀴즈: gawee-opt main() 구성
// ==========================================================================
//
// gawee-opt의 main() 함수를 재구성하라.
// Pass 등록, Dialect 등록, Bufferization 인터페이스 등록, MlirOptMain 호출을
// 올바른 순서로 작성해야 한다.
//

#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
// ... 기타 필요한 헤더

using namespace mlir;

namespace mlir::gawee {
std::unique_ptr<Pass> createGaweeToLinalgPass();
}

// ==========================================================================
// 문제 1: main() 작성
// ==========================================================================

int main(int argc, char **argv) {

  // (a) 단일 pass 등록: convert-gawee-to-linalg
  // CLI에서 --convert-gawee-to-linalg로 사용할 수 있도록 등록하라.
  _____<>(                                   // 어떤 클래스?
      "convert-gawee-to-linalg",
      "Lower Gawee dialect to Linalg dialect",
      [](_____ &pm) {                        // 어떤 타입?
        pm.addPass(gawee::createGaweeToLinalgPass());
      });

  // (b) 전체 파이프라인 등록: gawee-to-llvm
  // 다음 순서대로 pass를 추가하라:
  //   1. Gawee → Linalg
  //   2. tensor.empty → alloc_tensor (bufferize 준비)
  //   3. One-Shot Bufferize (bufferizeFunctionBoundaries = true)
  //   4. Linalg → Loops
  //   5. SCF → ControlFlow
  //   6. Arith → LLVM
  //   7. CF → LLVM
  //   8. MemRef → LLVM
  //   9. Func → LLVM
  //  10. Reconcile unrealized casts
  PassPipelineRegistration<>(
      "gawee-to-llvm",
      "Full pipeline: Gawee -> LLVM",
      [](OpPassManager &pm) {
        // 빈칸을 채우시오 (각 줄에 pm.addPass(...))
        pm.addPass(_____);  // Step 1
        pm.addPass(_____);  // Step 2
        // Step 3: Bufferize 옵션 설정
        bufferization::OneShotBufferizePassOptions bufOpts;
        bufOpts._____ = true;  // 어떤 옵션?
        pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
        pm.addPass(_____);  // Step 4
        pm.addPass(_____);  // Step 5
        pm.addPass(_____);  // Step 6
        pm.addPass(_____);  // Step 7
        pm.addPass(_____);  // Step 8
        pm.addPass(_____);  // Step 9
        pm.addPass(_____);  // Step 10
      });

  // ==========================================================================
  // 문제 2: Dialect 등록
  // ==========================================================================
  // 전체 파이프라인에서 사용하는 모든 dialect을 등록하라.

  DialectRegistry registry;

  // (c) 최소 9개의 dialect을 등록하시오
  registry.insert<_____>();  // 우리 dialect
  registry.insert<_____>();  // Linalg
  registry.insert<_____>();  // Arith
  registry.insert<_____>();  // Tensor
  registry.insert<_____>();  // Func
  registry.insert<_____>();  // SCF
  registry.insert<_____>();  // MemRef
  registry.insert<_____>();  // Bufferization
  registry.insert<_____>();  // LLVM

  // ==========================================================================
  // 문제 3: Bufferization 인터페이스 등록
  // ==========================================================================
  // One-Shot Bufferize가 각 dialect의 op을 어떻게 bufferize할지 알려준다.

  // (d) 4개의 인터페이스 등록 함수를 호출하시오
  _____::registerBufferizableOpInterfaceExternalModels(registry);
  _____::registerBufferizableOpInterfaceExternalModels(registry);
  _____::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::_____::registerBufferizableOpInterfaceExternalModels(registry);

  // ==========================================================================
  // 문제 4: MlirOptMain 호출
  // ==========================================================================

  // (e) MLIR 프레임워크의 표준 opt main 함수를 호출하라
  return mlir::asMainReturnCode(
      mlir::_____(argc, argv, "Gawee MLIR Optimizer\n", registry));
}


// ==========================================================================
// 문제 5: 개념 문제
// ==========================================================================
//
// (a) DialectRegistry에 dialect을 등록하지 않으면 어떤 오류가 발생하는가?
//     _____
//
// (b) bufferization 인터페이스를 등록하지 않으면 어떤 오류가 발생하는가?
//     _____
//
// (c) CMakeLists.txt에서 -fno-rtti를 설정하는 이유는?
//     _____
//
// (d) getDependentDialects의 역할은 무엇인가?
//     _____
//
// (e) PassPipelineRegistration과 PassRegistration의 차이는?
//     _____
//


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) PassPipelineRegistration, OpPassManager
  (b) Step 1:  gawee::createGaweeToLinalgPass()
      Step 2:  bufferization::createEmptyTensorToAllocTensorPass()
      Step 3:  bufferizeFunctionBoundaries
      Step 4:  createConvertLinalgToLoopsPass()
      Step 5:  createSCFToControlFlowPass()
      Step 6:  createArithToLLVMConversionPass()
      Step 7:  createConvertControlFlowToLLVMPass()
      Step 8:  createFinalizeMemRefToLLVMConversionPass()
      Step 9:  createConvertFuncToLLVMPass()
      Step 10: createReconcileUnrealizedCastsPass()

문제 2 (c):
  gawee::GaweeDialect
  linalg::LinalgDialect
  arith::ArithDialect
  tensor::TensorDialect
  func::FuncDialect
  scf::SCFDialect
  memref::MemRefDialect
  bufferization::BufferizationDialect
  LLVM::LLVMDialect

문제 3 (d):
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  linalg::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);

문제 4 (e):
  MlirOptMain

문제 5:
  (a) 해당 dialect의 op을 파싱할 때 "unregistered dialect" 오류가 발생한다.
      MLIR은 지연 로딩 방식이므로 등록하지 않은 dialect은 알 수 없다.
  (b) "don't know how to bufferize this op" 오류가 발생한다.
      One-Shot Bufferize가 각 op을 memref로 변환하는 방법을 모른다.
  (c) LLVM이 빌드 시 RTTI를 비활성화하기 때문이다.
      RTTI 설정이 다르면 링크 시 undefined symbol 오류가 발생한다.
  (d) pass가 실행될 때 생성하는 op의 dialect을 프레임워크에 알려준다.
      pass 매니저가 실행 전에 필요한 dialect을 미리 로드할 수 있도록 한다.
  (e) PassPipelineRegistration은 여러 pass를 묶어서 하나의 CLI 옵션으로 등록한다.
      PassRegistration은 단일 pass를 등록한다.

*/
