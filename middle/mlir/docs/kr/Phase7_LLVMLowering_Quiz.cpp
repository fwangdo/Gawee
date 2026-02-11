// ==========================================================================
// Phase 7 퀴즈: LLVM Lowering
// ==========================================================================
//
// Pass 순서, MemRef → LLVM 구조체 변환, unrealized_conversion_cast를 이해하는지 확인한다.
//

using namespace mlir;

// ==========================================================================
// 문제 1: Pass 순서
// ==========================================================================
//
// gawee-to-llvm 파이프라인의 LLVM lowering 부분(Step 5~7)에서,
// 다음 pass들을 올바른 순서로 나열하시오.
//
//   A. createConvertFuncToLLVMPass()
//   B. createReconcileUnrealizedCastsPass()
//   C. createSCFToControlFlowPass()
//   D. createArithToLLVMConversionPass()
//   E. createConvertControlFlowToLLVMPass()
//   F. createFinalizeMemRefToLLVMConversionPass()
//
// 올바른 순서: _____, _____, _____, _____, _____, _____  // (a)
//
// (b) 왜 FuncToLLVM이 MemRefToLLVM보다 뒤에 와야 하는가?
//     _____
//
// (c) ReconcileUnrealizedCasts가 반드시 마지막이어야 하는 이유는?
//     _____
//

// ==========================================================================
// 문제 2: SCF → ControlFlow 변환
// ==========================================================================
//
// 다음 SCF 코드가 ControlFlow(cf)로 어떻게 변환되는지 개략적으로 서술하시오.
//
// 변환 전:
//   scf.for %i = %lb to %ub step %step {
//     // body
//   }
//
// 변환 후 (빈칸 채우기):
//   cf.br ^_____(%lb)                    // (a) 어떤 블록?
//   ^header(%i):
//     %cond = arith._____ slt, %i, %ub   // (b) 비교 op
//     cf._____ %cond, ^body, ^exit        // (c) 어떤 분기?
//   ^body:
//     // body
//     %next = arith._____ %i, %step       // (d) 증가 op
//     cf.br ^_____(%next)                 // (e) 어떤 블록으로?
//   ^exit:
//

// ==========================================================================
// 문제 3: MemRef → LLVM 구조체
// ==========================================================================
//
// memref<1x64x112x112xf32>가 LLVM으로 변환될 때의 구조체를 채우시오.
//
// llvm.struct<{
//   _____,            // (a) 할당된 포인터
//   _____,            // (b) 정렬된 포인터 (실제 데이터 시작)
//   _____,            // (c) 오프셋
//   array<_____ x i64>, // (d) sizes 배열의 크기
//   array<_____ x i64>  // (e) strides 배열의 크기
// }>
//
// (f) 포인터가 2개인 이유는?
//     _____
//
// (g) strides의 값은 무엇인가? (shape이 [1, 64, 112, 112]일 때)
//     strides = [_____, _____, _____, _____]
//

// ==========================================================================
// 문제 4: memref.load의 LLVM 변환
// ==========================================================================
//
// memref.load %m[%i, %j, %k, %l] : memref<1x64x112x112xf32>
//
// 이것이 LLVM에서 어떻게 변환되는지 개략적으로 서술하시오.
//
// (a) 선형 인덱스 계산 공식:
//     linear_idx = offset + %i * _____ + %j * _____ + %k * _____ + %l * _____
//
// (b) 포인터 계산과 로드:
//     %ptr = llvm.getelementptr %base[%linear_idx]
//     %val = llvm._____  %ptr     // 어떤 op?
//

// ==========================================================================
// 문제 5: unrealized_conversion_cast
// ==========================================================================
//
// (a) unrealized_conversion_cast가 삽입되는 이유는?
//     _____
//
// (b) 다음 상황에서 cast가 어떻게 생기는지 설명하시오:
//     MemRefToLLVM pass 실행 후, FuncToLLVM 실행 전:
//     - memref.load → llvm.load로 변환됨 (llvm.struct 사용)
//     - 하지만 함수 시그니처는 아직 memref 타입
//     → unrealized_conversion_cast가 _____ 사이에 삽입됨
//
// (c) reconcile-unrealized-casts pass가 실패하면 어떤 의미인가?
//     _____
//

// ==========================================================================
// 문제 6: 전체 파이프라인
// ==========================================================================
//
// 다음 IR 변환 경로의 빈칸을 채우시오.
//
//   gawee.conv                         // Phase 3 입력
//     ↓ convert-gawee-to-linalg
//   _____.conv_2d_nchw_fchw            // (a) 어떤 dialect?
//     ↓ one-shot-bufferize
//   linalg.conv (_____ 타입)            // (b) tensor → ?
//     ↓ convert-linalg-to-loops
//   _____.for + arith ops              // (c) 어떤 dialect?
//     ↓ convert-scf-to-cf
//   _____.br + _____.cond_br           // (d) 어떤 dialect?
//     ↓ convert-to-llvm (여러 pass)
//   _____.br + _____.fadd              // (e) 최종 dialect?
//

// ==========================================================================
// 문제 7: mlir-translate
// ==========================================================================
//
// (a) LLVM dialect IR을 LLVM IR(.ll)로 변환하는 커맨드:
//     _____ --_____ output.mlir -o output.ll
//
// (b) LLVM IR을 오브젝트 파일로 컴파일하는 커맨드:
//     _____ output.ll -o output.o
//
// (c) 오브젝트 파일을 실행 파일로 링크하는 커맨드:
//     _____ output.o -o executable
//


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) C, D, E, F, A, B
      (SCFToControlFlow → ArithToLLVM → ControlFlowToLLVM → MemRefToLLVM → FuncToLLVM → ReconcileUnrealizedCasts)
  (b) FuncToLLVM이 함수 시그니처 전체를 LLVM 타입으로 변환하는데,
      이때 함수 본문의 memref op이 아직 변환되지 않으면 타입 불일치가 발생한다.
      MemRefToLLVM으로 memref op을 먼저 변환한 후, FuncToLLVM으로 함수를 래핑해야 한다.
  (c) 모든 dialect이 LLVM으로 변환된 후에만 unrealized_cast가 상쇄될 수 있다.
      중간에 실행하면 아직 변환되지 않은 타입 간의 cast가 남아 실패한다.

문제 2:
  (a) header
  (b) cmpi
  (c) cond_br
  (d) addi
  (e) header

문제 3:
  (a) ptr (allocated pointer)
  (b) ptr (aligned pointer)
  (c) i64 (offset)
  (d) 4
  (e) 4
  (f) allocated pointer는 malloc 반환값(해제 시 사용),
      aligned pointer는 메모리 정렬 후 실제 데이터 시작 주소.
  (g) strides = [802816, 12544, 112, 1]
      (64*112*112=802816, 112*112=12544, 112, 1)

문제 4:
  (a) linear_idx = offset + %i * 802816 + %j * 12544 + %k * 112 + %l * 1
  (b) llvm.load

문제 5:
  (a) 변환 과정에서 한 dialect은 이미 LLVM 타입을 사용하는데,
      아직 변환되지 않은 다른 dialect은 원래 타입(memref 등)을 기대하기 때문.
      임시로 타입 변환을 삽입하여 IR의 타입 일관성을 유지한다.
  (b) llvm.struct (LLVM 타입)와 memref 타입 사이에 삽입됨.
  (c) 어떤 op이 LLVM dialect으로 변환되지 못했다는 뜻이다.
      변환 패턴이 누락되었거나, dialect 등록이 빠져 있을 수 있다.

문제 6:
  (a) linalg
  (b) memref
  (c) scf
  (d) cf.br + cf.cond_br
  (e) llvm.br + llvm.fadd

문제 7:
  (a) mlir-translate --mlir-to-llvmir output.mlir -o output.ll
  (b) llc output.ll -o output.o
  (c) clang output.o -o executable

*/
