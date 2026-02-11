// ==========================================================================
// Phase 5 퀴즈: Bufferization과 Linalg → Loops
// ==========================================================================
//
// 파이프라인 구성과 개념 문제로 구성되어 있다.
// Bufferization의 의미와 tensor/memref 차이를 이해하고 있는지 확인한다.
//

using namespace mlir;

// ==========================================================================
// 문제 1: tensor vs memref
// ==========================================================================
//
// 다음 표의 빈칸을 채우시오.
//
// |           | tensor              | memref              |
// |-----------|---------------------|---------------------|
// | 의미론    | _____ (a)           | _____ (b)           |
// | 불변성    | _____ (c)           | _____ (d)           |
// | C++ 비유  | _____ (e)           | _____ (f)           |
// | 용도      | _____ (g)           | _____ (h)           |
//

// ==========================================================================
// 문제 2: 파이프라인 구성
// ==========================================================================
//
// gawee-to-loops 파이프라인을 올바른 순서로 작성하시오.
// 아래 3개의 pass를 올바른 순서로 배치하라.
//
//   A. createConvertLinalgToLoopsPass()
//   B. bufferization::createOneShotBufferizePass(bufOpts)
//   C. gawee::createGaweeToLinalgPass()
//
// 올바른 순서: _____, _____, _____  // (a) C, B, A 같은 형태로

// ==========================================================================
// 문제 3: Bufferization 설정
// ==========================================================================
//
// 다음 코드의 빈칸을 채우시오.
//
//   bufferization::OneShotBufferizePassOptions bufOpts;
//   bufOpts._____ = true;  // (a) 함수 경계도 bufferize하는 옵션
//   pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));
//
// (b) 위 옵션이 true일 때, 함수 시그니처는 어떻게 변하는가?
//   변환 전: func @forward(%arg0: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32>
//   변환 후: func @forward(%arg0: _____<1x3x224x224xf32>) -> _____<1x1000xf32>
//

// ==========================================================================
// 문제 4: Destination-Passing과 Bufferization의 관계
// ==========================================================================
//
// 다음 tensor 코드가 bufferize된 후 어떻게 변하는지 빈칸을 채우시오.
//
// 변환 전:
//   %empty = tensor.empty [64] : tensor<64xf32>
//   %filled = linalg.fill ins(%zero) outs(%empty) -> tensor<64xf32>
//   %result = linalg.add ins(%a, %b) outs(%filled) -> tensor<64xf32>
//
// 변환 후:
//   %alloc = _____._____ () : memref<64xf32>       // (a) 메모리 할당 op
//   linalg.fill ins(%zero) outs(_____)              // (b) 어떤 값?
//   linalg.add ins(%a_memref, %b_memref) outs(____) // (c) 어떤 값?
//
// (d) destination-passing style이 bufferization에 유리한 이유를 서술하시오.
//     _____

// ==========================================================================
// 문제 5: Linalg → Loops 변환
// ==========================================================================
//
// 다음 linalg.generic이 SCF 루프로 변환된 후 모습을 빈칸으로 채우시오.
//
// 변환 전:
//   linalg.generic {
//     indexing_maps = [identity_4d, identity_4d],
//     iterator_types = ["parallel", "parallel", "parallel", "parallel"]
//   } ins(%input : memref<1x64x112x112xf32>)
//     outs(%output : memref<1x64x112x112xf32>) {
//     ^bb0(%in: f32, %out: f32):
//       %zero = arith.constant 0.0 : f32
//       %result = arith.maximumf %in, %zero : f32
//       linalg.yield %result : f32
//   }
//
// 변환 후 (개략적):
//   _____.for %n = 0 to 1 {                           // (a) 어떤 dialect?
//     _____.for %c = 0 to 64 {
//       _____.for %h = 0 to 112 {
//         _____.for %w = 0 to 112 {
//           %val = _____.load %input[%n, %c, %h, %w]  // (b) 어떤 op?
//           %zero = arith.constant 0.0 : f32
//           %result = arith.maximumf %val, %zero : f32
//           _____.store %result, %output[%n, %c, %h, %w]  // (c) 어떤 op?
//         }
//       }
//     }
//   }
//

// ==========================================================================
// 문제 6: 개념 문제
// ==========================================================================
//
// (a) tensor를 먼저 사용하고 나중에 memref로 변환하는 이유는?
//     _____
//
// (b) One-Shot Bufferize의 "One-Shot"이 의미하는 것은?
//     _____
//
// (c) iterator_types에서 "parallel"과 "reduction"의 차이는?
//     _____
//
// (d) SCF dialect의 full name과 특징은?
//     _____
//
// (e) createEmptyTensorToAllocTensorPass()의 역할은?
//     _____
//
// (f) Bufferization 인터페이스 등록이 필요한 이유는?
//     _____
//


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) 값 (value semantics)
  (b) 참조 (reference semantics)
  (c) 불변 (immutable, SSA)
  (d) 가변 (mutable, 메모리에 저장)
  (e) const int x = 5 (불변 값)
  (f) int *ptr (포인터, 메모리 참조)
  (g) 고수준 최적화 (fusion, tiling 등)
  (h) 실제 메모리 접근, 코드 생성

문제 2:
  C, B, A (Gawee→Linalg → Bufferize → Linalg→Loops)

문제 3:
  (a) bufferizeFunctionBoundaries
  (b) memref<1x3x224x224xf32> → memref<1x1000xf32>

문제 4:
  (a) memref.alloc
  (b) %alloc
  (c) %alloc
  (d) outs 텐서가 그대로 출력 memref가 된다.
      별도 메모리 할당 없이 outs에 지정한 텐서를 memref로 변환하면
      in-place 연산이 가능하다. tensor.empty → memref.alloc 매핑이 직관적이다.

문제 5:
  (a) scf
  (b) memref.load
  (c) memref.store

문제 6:
  (a) tensor는 SSA 형태라 의존성 분석과 최적화(fusion, tiling)가 쉽다.
      memref는 alias 분석이 필요하고 메모리 관리가 복잡하다.
      고수준에서 최적화 후 memref로 변환하는 것이 효율적이다.
  (b) 전체 프로그램을 한 번에 분석하여 최적의 buffer 할당을 결정한다.
      반복적(iterative)이 아닌 단일 패스(one-shot)로 처리한다.
  (c) parallel: 루프 간 데이터 의존성 없음. 독립적으로 병렬 실행 가능.
      reduction: 루프 간 데이터 의존성 있음 (예: sum, max). 누적 변수 필요.
  (d) Structured Control Flow. for, while, if 같은 구조화된 제어 흐름.
      CFG(기본 블록 + 분기)보다 높은 수준이라 분석/최적화가 쉽다.
  (e) tensor.empty를 bufferization.alloc_tensor로 변환한다.
      One-Shot Bufferize가 alloc_tensor를 인식하여 memref.alloc으로 변환할 수 있게 한다.
  (f) One-Shot Bufferize가 각 dialect의 op을 어떻게 memref 연산으로 바꿀지 알아야 한다.
      인터페이스 없이는 "이 op을 bufferize하는 방법을 모른다"는 오류가 발생한다.

*/
