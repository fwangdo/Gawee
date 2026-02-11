// ==========================================================================
// Phase 2 퀴즈: TableGen과 Dialect 정의
// ==========================================================================
//
// 이 파일은 C++ 주석 안에 TableGen(.td) 내용을 담고 있다.
// 빈칸(_____)을 채워서 완성하라.
//
// 목표: .td 파일 문법, 속성 타입, tblgen 커맨드를 정확히 이해하고 있는지 확인
//

// ==========================================================================
// 문제 1: Dialect 정의 (GaweeDialect.td)
// ==========================================================================
//
// 다음 TableGen 코드의 빈칸을 채우시오.
//
//   include "_____"              // (a) MLIR 기본 정의를 포함하는 .td 파일
//
//   def Gawee_Dialect : _____ {  // (b) 어떤 TableGen 클래스를 상속?
//     let name = "_____";        // (c) IR에서 gawee.conv처럼 보이려면?
//     let cppNamespace = "::mlir::gawee";
//     let _____ = 0;             // (d) bytecode 속성 방식 비활성화
//   }
//
//   class Gawee_Op<string mnemonic, list<_____> traits = []> :  // (e) traits의 타입
//       Op<_____, mnemonic, traits>;                             // (f) 어떤 dialect에 속하는지
//

// ==========================================================================
// 문제 2: Op 정의 (GaweeOps.td)
// ==========================================================================
//
// Conv op을 정의하시오. 빈칸을 채우라.
//
//   def Gawee_ConvOp : Gawee_Op<"_____", []> {   // (a) op mnemonic
//     let summary = "2D convolution";
//     let arguments = (___                         // (b) 입력을 정의하는 키워드
//       AnyTensor:$input,
//       AnyTensor:$weight,
//       AnyTensor:$bias,
//       _____:$strides,                            // (c) int64 배열 속성 타입
//       _____:$padding,                            // (d) 같은 타입
//       _____:$dilation                            // (e) 같은 타입
//     );
//     let results = (_____ AnyTensor:$output);     // (f) 출력을 정의하는 키워드
//   }
//

// ==========================================================================
// 문제 3: 다양한 속성 타입
// ==========================================================================
//
// 다음 op 정의에서 올바른 속성 타입을 쓰시오.
//
//   def Gawee_BatchNormOp : Gawee_Op<"batch_norm", []> {
//     let arguments = (ins
//       AnyTensor:$input,
//       AnyTensor:$weight,
//       AnyTensor:$bias,
//       _____:$affine,        // (a) 참/거짓 값
//       AnyTensor:$runningMean,
//       AnyTensor:$runningVar,
//       _____:$eps            // (b) 실수(double) 값
//     );
//     let results = (outs AnyTensor:$output);
//   }
//
//   def Gawee_FlattenOp : Gawee_Op<"flatten", []> {
//     let arguments = (ins
//       AnyTensor:$input,
//       _____:$startDim,      // (c) 단일 정수 값
//       _____:$endDim         // (d) 단일 정수 값
//     );
//     let results = (outs AnyTensor:$output);
//   }
//

// ==========================================================================
// 문제 4: build.sh의 tblgen 커맨드
// ==========================================================================
//
// 다음 빈칸을 채우시오.
//
// (a) Dialect 클래스 선언(헤더)을 생성하는 tblgen 플래그:
//     $TBLGEN _____ include/Gawee/GaweeDialect.td -o GaweeDialect.h.inc
//
// (b) Op 클래스 정의(구현)를 생성하는 tblgen 플래그:
//     $TBLGEN _____ include/Gawee/GaweeOps.td -o GaweeOps.cpp.inc
//
// (c) -I 플래그의 역할은?
//     _____
//
// (d) .inc 파일을 사용하려면 .cpp에서 어떻게 해야 하는가?
//     _____
//

// ==========================================================================
// 문제 5: .inc 파일 사용법
// ==========================================================================
//
// GaweeDialect.cpp에서 .inc 파일을 포함하는 코드를 완성하시오.
//
//   #include "Gawee/GaweeDialect.h"
//
//   #include "Gawee/generated/_____"          // (a) Dialect 구현 .inc 파일
//
//   #define _____                               // (b) Op 구현을 활성화하는 매크로
//   #include "Gawee/generated/_____"          // (c) Op 구현 .inc 파일
//

// ==========================================================================
// 문제 6: $이름의 의미
// ==========================================================================
//
// 다음 TableGen 정의가 있을 때:
//
//   def Gawee_ConvOp : Gawee_Op<"conv", []> {
//     let arguments = (ins AnyTensor:$input, DenseI64ArrayAttr:$strides);
//     let results = (outs AnyTensor:$output);
//   }
//
// (a) $input에 대해 자동 생성되는 C++ getter 메서드 이름: _____
// (b) $strides에 대해 Attribute 객체를 반환하는 getter: _____
// (c) $output에 대해 자동 생성되는 getter: _____
// (d) op의 C++ 클래스 이름: _____
// (e) IR에서 이 op의 이름: _____
//


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) "mlir/IR/OpBase.td"
  (b) Dialect
  (c) "gawee"
  (d) usePropertiesForAttributes
  (e) Trait
  (f) Gawee_Dialect

문제 2:
  (a) "conv"
  (b) ins
  (c) DenseI64ArrayAttr
  (d) DenseI64ArrayAttr
  (e) DenseI64ArrayAttr
  (f) outs

문제 3:
  (a) BoolAttr
  (b) F64Attr
  (c) I64Attr
  (d) I64Attr

문제 4:
  (a) --gen-dialect-decls
  (b) --gen-op-defs
  (c) include 구문(예: include "mlir/IR/OpBase.td")의 검색 경로를 지정한다
  (d) #include로 .cpp/.h 안에 삽입한다 (.inc는 독립 컴파일 불가)

문제 5:
  (a) GaweeDialect.cpp.inc
  (b) GET_OP_CLASSES
  (c) GaweeOps.cpp.inc

문제 6:
  (a) getInput()
  (b) getStridesAttr()
  (c) getOutput()
  (d) gawee::ConvOp (namespace::mlir::gawee)
  (e) gawee.conv

*/
