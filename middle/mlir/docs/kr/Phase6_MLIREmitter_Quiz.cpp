// ==========================================================================
// Phase 6 퀴즈: MLIREmitter — emitConv, emitLinear 재구현
// ==========================================================================
//
// JSON에서 Gawee MLIR을 생성하는 emitter 함수를 재구현하라.
// valueMap, OpBuilder, llvm::json API의 사용법을 이해하고 있는지 확인한다.
//

#include "Emit/MLIREmitter.h"
#include "Gawee/GaweeDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::gawee;

// ==========================================================================
// 문제 1: emit() 함수의 2-Pass 구조
// ==========================================================================
//
// emit() 함수는 왜 2-Pass 구조인가?
//
// Pass 1의 역할: _____  (a)
// Pass 2의 역할: _____  (b)
//
// 왜 1-Pass로 처리할 수 없는가?
//   _____  (c)
//

// ==========================================================================
// 문제 2: emitConv 재구현
// ==========================================================================
//
// 빈칸을 채워 emitConv를 완성하시오.
//
bool MLIREmitter_emitConv(
    MLIREmitter *self,
    OpBuilder &builder,
    std::unordered_map<std::string, Value> &valueMap,
    const llvm::json::Object &node,
    const llvm::json::Object &values) {

  auto loc = builder.getUnknownLoc();

  // (a) JSON에서 input 이름 읽기
  const auto *inputs = node._____("inputs");     // 어떤 메서드?
  auto inputName = (*inputs)[0]._____();          // 문자열로 변환
  Value input = valueMap[inputName->str()];

  // (b) JSON에서 output shape 읽어서 결과 타입 만들기
  const auto *outputs = node._____("outputs");    // 어떤 메서드?
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values._____(*outputName);  // Object 가져오기
  // shape 배열에서 RankedTensorType 생성 (parseShape 호출)

  // (c) weight/bias 가져오기 — 왜 valueMap에서 찾는가?
  // _____  (설명)
  auto nodeName = node.getString("name");
  std::string weightName = nodeName->str() + "_____";  // (d) 접미사
  std::string biasName = nodeName->str() + "_____";    // (e) 접미사
  Value weight = valueMap[weightName];
  Value bias = valueMap[biasName];

  // (f) JSON에서 conv 속성 읽기
  const auto *attrs = node._____("attrs");  // 어떤 메서드?

  // stride, padding, dilation 배열 읽기
  SmallVector<int64_t> strides;
  if (const auto *arr = attrs->_____("stride")) {  // (g) 배열 메서드
    for (const auto &v : *arr) {
      if (auto i = v._____()) strides.push_back(*i);  // (h) 정수 변환
    }
  }
  // padding, dilation도 동일한 방식

  // (i) Gawee ConvOp 생성
  auto convOp = builder.create<_____>(  // 어떤 op?
      loc, /*resultType*/ nullptr, input, weight, bias,
      builder._____( strides),    // (j) DenseI64ArrayAttr 생성 메서드
      builder.getDenseI64ArrayAttr(/*padding*/{}),
      builder.getDenseI64ArrayAttr(/*dilation*/{}));

  // (k) 출력을 valueMap에 등록
  valueMap[outputName->str()] = convOp._____();  // 어떤 메서드?

  return true;
}


// ==========================================================================
// 문제 3: emitLinear 재구현
// ==========================================================================
//
// Linear(MatMul) emit 함수의 빈칸을 채우시오.
// 주의: JSON의 op_type은 "MatMul"이지만, 생성하는 MLIR op은 gawee.linear이다.
//
bool MLIREmitter_emitLinear(
    MLIREmitter *self,
    OpBuilder &builder,
    std::unordered_map<std::string, Value> &valueMap,
    const llvm::json::Object &node,
    const llvm::json::Object &values) {

  auto loc = builder.getUnknownLoc();

  // (a) input 가져오기 (Conv와 동일한 패턴)
  const auto *inputs = node.getArray("inputs");
  auto inputName = (*inputs)[0].getAsString();
  Value input = valueMap[inputName->str()];

  // (b) output type
  const auto *outputs = node.getArray("outputs");
  auto outputName = (*outputs)[0].getAsString();
  const auto *outputInfo = values.getObject(*outputName);
  // resultType = parseShape(...)

  // (c) weight/bias — Conv와 동일한 패턴
  auto nodeName = node._____("name");        // 어떤 메서드?
  Value weight = valueMap[nodeName->str() + "_weight"];
  Value bias = valueMap[nodeName->str() + "_bias"];

  // (d) LinearOp 생성
  auto linearOp = builder.create<_____>(  // 어떤 op?
      loc, /*resultType*/ nullptr, input, weight, bias);

  // (e) 출력 등록
  valueMap[_____] = linearOp.getResult();  // 어떤 키?

  return true;
}


// ==========================================================================
// 문제 4: parseShape 구현
// ==========================================================================
//
// JSON의 shape 배열을 RankedTensorType으로 변환하는 함수를 작성하시오.
//
// RankedTensorType parseShape(MLIRContext *ctx, const llvm::json::Array *shape) {
//   SmallVector<int64_t> dims;
//   for (const auto &dim : *shape) {
//     dims.push_back(*dim._____());  // (a) 정수 변환 메서드
//   }
//   return RankedTensorType::get(dims, _____::get(ctx));  // (b) 기본 원소 타입
// }
//

// ==========================================================================
// 문제 5: valueMap의 역할
// ==========================================================================
//
// 다음 그래프가 있을 때 valueMap의 상태 변화를 추적하시오.
//
// 그래프:
//   input → Conv(conv1) → ReLU(relu1) → output
//
// 함수 시그니처:
//   func @forward(%input: tensor<1x3x224x224xf32>,
//                 %conv1_weight: tensor<64x3x7x7xf32>,
//                 %conv1_bias: tensor<64xf32>)
//
// (a) 함수 인자 매핑 후 valueMap:
//   "input"        → _____
//   "conv1_weight" → _____
//   "conv1_bias"   → _____
//
// (b) emitConv("conv1") 실행 후 추가되는 항목:
//   "_____" → _____
//
// (c) emitRelu("relu1") 실행 후 추가되는 항목:
//   "_____" → _____
//

// ==========================================================================
// 문제 6: gawee-translate vs gawee-opt
// ==========================================================================
//
// (a) gawee-translate의 입력과 출력은?
//     입력: _____     출력: _____
//
// (b) gawee-opt의 입력과 출력은?
//     입력: _____     출력: _____
//
// (c) gawee-translate에서 dialect을 로드하는 방법:
//     context._____<gawee::GaweeDialect>();  // 어떤 메서드?
//
// (d) gawee-opt에서 dialect을 등록하는 방법:
//     registry._____<gawee::GaweeDialect>();  // 어떤 메서드?
//
// (e) loadDialect과 insert의 차이:
//     _____
//


// ==========================================================================
// 정답
// ==========================================================================

/* 정답

문제 1:
  (a) weight/bias 텐서 정보를 수집하여 weightArgs에 저장
  (b) 각 노드를 순서대로 Gawee op으로 변환 (emitNode 호출)
  (c) 함수 시그니처를 만들려면 모든 weight의 shape을 먼저 알아야 하기 때문.
      weight는 함수 인자로 전달되므로 함수 생성 시점에 타입이 필요하다.

문제 2:
  (a) getArray, getAsString
  (b) getArray, getObject
  (c) weight/bias는 first pass에서 수집되어 함수 인자로 추가되었고,
      valueMap에 이미 매핑되어 있기 때문.
  (d) "_weight"
  (e) "_bias"
  (f) getObject
  (g) getArray
  (h) getAsInteger
  (i) ConvOp
  (j) getDenseI64ArrayAttr
  (k) getResult

문제 3:
  (c) getString
  (d) LinearOp
  (e) outputName->str()

문제 4:
  (a) getAsInteger
  (b) Float32Type

문제 5:
  (a) "input"        → entryBlock->getArgument(0)
      "conv1_weight" → entryBlock->getArgument(1)
      "conv1_bias"   → entryBlock->getArgument(2)
  (b) "conv1_output" (또는 JSON의 outputs[0] 이름) → convOp.getResult()
  (c) "relu1_output" (또는 JSON의 outputs[0] 이름) → reluOp.getResult()

문제 6:
  (a) 입력: JSON 파일, 출력: Gawee MLIR 텍스트
  (b) 입력: MLIR 텍스트, 출력: 변환된 MLIR 텍스트
  (c) loadDialect
  (d) insert
  (e) loadDialect은 즉시 dialect을 로드한다 (eager).
      insert는 registry에 등록만 하고 필요할 때 로드한다 (lazy).
      gawee-translate는 직접 op을 생성하므로 즉시 필요.
      gawee-opt는 MlirOptMain이 필요시 로드하므로 lazy 등록.

*/
