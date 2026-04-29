// MatMul lowering quiz
//
// 목표:
// 1. ONNX MatMul과 Gemm의 차이를 정확히 구분한다.
// 2. linalg.generic 기반 matmul lowering의 loop / indexing map을 복원한다.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;

namespace {

// 문제 1
// 아래 빈칸을 채워라.
//
// ONNX MatMul의 일반 형태는
//   [..., __, __] x [..., __, __] -> [..., __, __]
//
// 답:
//   lhs 마지막 2축: (____, ____)
//   rhs 마지막 2축: (____, ____)
//   output 마지막 2축: (____, ____)

// 문제 2
// iterator type을 채워라.
//
// batch dims + M + N + K loop를 가진 generic matmul에서
//   batch dims: __________
//   M: __________
//   N: __________
//   K: __________

// 문제 3
// rhs indexing map의 마지막 두 식을 채워라.
//
// loop dims 순서가 [batch..., m, n, k] 라면
// rhs는 [..., k, n] 이므로 마지막 두 식은
//   1. __________________
//   2. __________________

// 문제 4
// 아래 skeleton의 TODO를 채워라.
static Value buildQuizMatmul(ConversionPatternRewriter &rewriter, Location loc,
                             Value lhs, Value rhs, RankedTensorType outputType,
                             int64_t batchRank, Type elementType) {
  Value empty = tensor::EmptyOp::create(
      rewriter, loc, outputType.getShape(), elementType,
      ValueRange{});
  Value zero = arith::ConstantOp::create(
      rewriter, loc, elementType, rewriter.getZeroAttr(elementType));
  Value init = linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);

  SmallVector<AffineMap> indexingMaps; // TODO: lhs / rhs / out map 채우기
  SmallVector<utils::IteratorType> iteratorTypes;
  // TODO: parallel / reduction 구성하기

  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, TypeRange{outputType}, ValueRange{lhs, rhs},
      ValueRange{init}, indexingMaps, iteratorTypes,
      [&](OpBuilder &builder, Location bodyLoc, ValueRange args) {
        // args[0] = lhs scalar
        // args[1] = rhs scalar
        // args[2] = accumulator
        Value product = arith::MulFOp::create(builder, bodyLoc, args[0], args[1]);
        Value result = arith::AddFOp::create(builder, bodyLoc, product, args[2]);
        linalg::YieldOp::create(builder, bodyLoc, result);
      });
  return genericOp.getResult(0);
}

} // namespace
