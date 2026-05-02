#ifndef GAWEE_OPS_TD
#define GAWEE_OPS_TD

include "GaweeDialect.td"
include "mlir/IR/OpBase.td"

def Gawee_ConvOp : Gawee_Op<"conv", []> {
  let summary = "2D convolution";
  let arguments = (ins 
    AnyTensor:$input, 
    AnyTensor:$weight,
    AnyTensor:$bias, 
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_ReluOp : Gawee_Op<"relu", []> {
  let summary = "ReLU activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

def Gawee_AddOp : Gawee_Op<"add", []> {
  let summary = "Elementwise add";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
}

def Gawee_BatchNormOp : Gawee_Op<"batch_norm", []> {
  let summary = "Batch normalization"; 
  let arguments = (ins
    AnyTensor:$input, 
    AnyTensor:$weight, // scale
    AnyTensor:$bias, // bias 
    BoolAttr:$affine, // is it affine or not.  
    AnyTensor:$runningMean, 
    AnyTensor:$runningVar, 
    F64Attr:$eps 
  ); 
  let results = (outs AnyTensor:$output); 
}

// added for resnet. 
def Gawee_MaxPoolOp : Gawee_Op<"max_pool", []> {
  let summary = "Max pooling"; 
  let arguments = (ins
    AnyTensor:$input, 
    DenseI64ArrayAttr:$kernelSize, 
    DenseI64ArrayAttr:$strides, 
    DenseI64ArrayAttr:$padding, 
    DenseI64ArrayAttr:$dilation, 
    BoolAttr:$ceilMode 
  );
  let results = (outs AnyTensor:$output); 
}

def Gawee_AveragePoolOp : Gawee_Op<"average_pool", []> {
  let summary = "Average pooling";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernelSize,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    BoolAttr:$ceilMode,
    BoolAttr:$countIncludePad
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_AdAvgPoolOp : Gawee_Op<"ad_avg_pool", []> {
  let summary = "adaptive average pooling"; 

  let arguments = (ins
    AnyTensor:$input, 
    DenseI64ArrayAttr:$outputSize
  ); 
  let results = (outs AnyTensor:$output); 
}

def Gawee_FlattenOp : Gawee_Op<"flatten", []> {
  let summary = "flatten operation"; 

  let arguments = (ins
    AnyTensor:$input, 
    I64Attr:$startDim, 
    I64Attr:$endDim 
  ); 
  let results = (outs AnyTensor:$output); 
}

def Gawee_LinearOp : Gawee_Op<"linear", []> {
  let summary = "Linear layer";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_CatOp : Gawee_Op<"cat", []> {
  let summary = "cat operation";

  let arguments = (ins
    Variadic<AnyTensor>:$inputs,
    I64Attr:$axis 
  );
  let results = (outs AnyTensor:$output); 
}

def Gawee_GlobalAveragePoolOp : Gawee_Op<"global_average_pool", []> {
  let summary = "Global average pooling";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
}

// -------------------------------------------------------------------------
// ONNX normalized path scaffolds
//
// Why these are here:
//   - current JSON path defined higher-level ops first
//   - normalized ONNX path will need more primitive tensor/arithmetic ops
//   - the target 3 ONNX models will eventually need a subset of these
//
// Implementation rule:
//   1. add emitter support in ONNXMLIREmitter.cpp
//   2. then add lowering in GaweeToLinalg.cpp
//   3. only then use the op in end-to-end tests
//
// Suggested implementation order:
//   - arithmetic: sub, mul, div
//   - shape/layout: reshape, transpose, squeeze, unsqueeze
//   - reduction: reduce_mean, reduce_sum
//   - activation/probability: softmax, sqrt, tanh, sigmoid, erf
//   - mask/control helpers: equal, where
//   - tensor plumbing: expand, slice, cast
//
// Notes:
//   - Concat is already represented as gawee.cat
//   - Conv / Relu / Add / Linear are already defined above
// -------------------------------------------------------------------------

def Gawee_SubOp : Gawee_Op<"sub", []> {
  let summary = "Elementwise subtraction";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Sub -> gawee.sub
  // - lowering: decide whether linalg.generic is enough or if a direct op exists
  // - reference: Gawee_AddOp / AddOpLowering
}

def Gawee_MulOp : Gawee_Op<"mul", []> {
  let summary = "Elementwise multiplication";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Mul -> gawee.mul
  // - lowering: linalg.generic is the likely first target
  // - check whether broadcast must already be normalized before this stage
  // - reference: Gawee_AddOp / AddOpLowering
}

def Gawee_DivOp : Gawee_Op<"div", []> {
  let summary = "Elementwise division";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Div -> gawee.div
  // - lowering: likely linalg.generic + arith::DivFOp
  // - reference: AdAvgPool lowering uses DivFOp in a generic body
}

def Gawee_MatMulOp : Gawee_Op<"matmul", []> {
  let summary = "General ONNX-style MatMul";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_ReduceMeanOp : Gawee_Op<"reduce_mean", []> {
  let summary = "ReduceMean over selected axes";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$axes,
    BoolAttr:$keepDims
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: parse ONNX axes / keepdims carefully
  // - lowering: decide whether to lower directly or decompose to ReduceSum + scalar divide
  // - reference: AdAvgPool lowering shows a sum-then-div pattern
}

def Gawee_ReduceSumOp : Gawee_Op<"reduce_sum", []> {
  let summary = "ReduceSum over selected axes";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$axes,
    BoolAttr:$keepDims
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX ReduceSum -> gawee.reduce_sum
  // - lowering: think first about the exact linalg form you want
  // - reference: pool-style sum lowering in AdAvgPool
}

def Gawee_ReshapeOp : Gawee_Op<"reshape", []> {
  let summary = "Tensor reshape";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$shape
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: decide whether shape comes from initializer or runtime tensor
  // - lowering: likely tensor::ExpandShapeOp / tensor::CollapseShapeOp / tensor::ReshapeOp family
  // - reference: current lowering file already includes ReshapeOpsUtils.h
}

def Gawee_TransposeOp : Gawee_Op<"transpose", []> {
  let summary = "Tensor transpose";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$perm
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: parse ONNX perm attr, or default reverse order if absent
  // - lowering: likely linalg.generic or tensor-level transpose path
  // - reference: think about how Conv / Linear lowering already assumes certain layouts
}

def Gawee_SqueezeOp : Gawee_Op<"squeeze", []> {
  let summary = "Remove unit dimensions";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$axes
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Squeeze in newer opsets may take axes as input tensor
  // - lowering: likely reshape-like, not arithmetic
  // - reference: group with Reshape/Unsqueeze work
}

def Gawee_UnsqueezeOp : Gawee_Op<"unsqueeze", []> {
  let summary = "Insert unit dimensions";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$axes
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Unsqueeze often carries axes as tensor input, not attr
  // - lowering: likely reshape-like, not arithmetic
  // - reference: pair with Squeeze and Reshape handling
}

def Gawee_ShapeOp : Gawee_Op<"shape", []> {
  let summary = "Read tensor shape";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$start,
    I64Attr:$end
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Shape may omit start/end and needs default normalization
  // - lowering: usually becomes shape/tensor dimension materialization
}

def Gawee_SoftmaxOp : Gawee_Op<"softmax", []> {
  let summary = "Softmax along one axis";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: parse ONNX axis and normalize negative axis
  // - lowering: decide whether you want a direct high-level op for now,
  //   or a decomposition path later
  // - reference: target models in NLP will likely need this
}

def Gawee_SqrtOp : Gawee_Op<"sqrt", []> {
  let summary = "Elementwise square root";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Sqrt -> gawee.sqrt
  // - lowering: likely linalg.generic + arith::SqrtOp or math op path
}

def Gawee_TanhOp : Gawee_Op<"tanh", []> {
  let summary = "Elementwise tanh";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Tanh -> gawee.tanh
  // - lowering: likely linalg.generic + math/arith path depending available dialect
}

def Gawee_SigmoidOp : Gawee_Op<"sigmoid", []> {
  let summary = "Elementwise sigmoid";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Sigmoid -> gawee.sigmoid
  // - lowering: decide whether to keep as high-level op first or decompose
}

def Gawee_HardSigmoidOp : Gawee_Op<"hard_sigmoid", []> {
  let summary = "Elementwise hard sigmoid";
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha,
    F64Attr:$beta
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: parse ONNX alpha/beta attrs with ONNX defaults
  // - lowering: likely clamp(alpha * x + beta, 0, 1)
}

def Gawee_HardSwishOp : Gawee_Op<"hard_swish", []> {
  let summary = "Elementwise hard swish";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX HardSwish -> gawee.hard_swish
  // - lowering: likely x * hard_sigmoid(x)
}

def Gawee_LeakyReluOp : Gawee_Op<"leaky_relu", []> {
  let summary = "Elementwise leaky ReLU";
  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$alpha
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: parse ONNX alpha attr with default normalization
  // - lowering: select(x >= 0, x, alpha * x)
}

def Gawee_GeluOp : Gawee_Op<"gelu", []> {
  let summary = "Elementwise GELU";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Gelu -> gawee.gelu
  // - lowering: exact vs approximate contract must be decided first
}

def Gawee_ErfOp : Gawee_Op<"erf", []> {
  let summary = "Elementwise error function";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Erf -> gawee.erf
  // - lowering: confirm which MLIR dialect/op to use before coding
  // - reference: check how GELU would eventually be represented
}

def Gawee_EqualOp : Gawee_Op<"equal", []> {
  let summary = "Elementwise equality comparison";
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Equal -> gawee.equal
  // - lowering: think about bool tensor result type and later Where use
}

def Gawee_WhereOp : Gawee_Op<"where", []> {
  let summary = "Elementwise select";
  let arguments = (ins
    AnyTensor:$condition,
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Where -> gawee.where
  // - lowering: likely linalg.generic with select
  // - reference: this matters for mask-style graph patterns
}

def Gawee_MaxOp : Gawee_Op<"max", []> {
  let summary = "Elementwise max";
  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Max is variadic
  // - lowering: reduce pairwise with max primitive
}

def Gawee_MinOp : Gawee_Op<"min", []> {
  let summary = "Elementwise min";
  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Min is variadic
  // - lowering: reduce pairwise with min primitive
}

def Gawee_ExpandOp : Gawee_Op<"expand", []> {
  let summary = "Broadcast tensor to target shape";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$shape
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: distinguish shape tensor from ordinary runtime tensor
  // - lowering: first decide whether expand survives to middle-end at all
}

def Gawee_GatherOp : Gawee_Op<"gather", []> {
  let summary = "Gather slices from data along one axis";
  let arguments = (ins
    AnyTensor:$data,
    AnyTensor:$indices,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_GatherElementsOp : Gawee_Op<"gather_elements", []> {
  let summary = "Gather elements using per-position indices";
  let arguments = (ins
    AnyTensor:$data,
    AnyTensor:$indices,
    I64Attr:$axis
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_RangeOp : Gawee_Op<"range", []> {
  let summary = "Construct a 1-D arithmetic progression";
  let arguments = (ins
    AnyTensor:$start,
    AnyTensor:$limit,
    AnyTensor:$delta
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_ResizeOp : Gawee_Op<"resize", []> {
  let summary = "Resize tensor with preserved ONNX mode semantics";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$scales,
    StrAttr:$mode,
    StrAttr:$coordinateTransformationMode,
    StrAttr:$nearestMode
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_SplitOp : Gawee_Op<"split", []> {
  let summary = "Split tensor into multiple outputs along one axis";
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$splitSizes,
    I64Attr:$axis
  );
  let results = (outs Variadic<AnyTensor>:$outputs);
}

def Gawee_SliceOp : Gawee_Op<"slice", []> {
  let summary = "Tensor slice";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$starts,
    AnyTensor:$ends,
    AnyTensor:$axes,
    AnyTensor:$steps
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Slice is input-driven in newer opsets
  // - lowering: clarify whether all five tensors are required in your contract
  // - reference: ONNX contract first, code second
}

def Gawee_PadOp : Gawee_Op<"pad", []> {
  let summary = "Tensor padding";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$pads,
    AnyTensor:$constantValue,
    StrAttr:$mode
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Pad uses pads/value as inputs in newer opsets
  // - lowering: decide whether padding survives as a dedicated high-level op
}

def Gawee_TileOp : Gawee_Op<"tile", []> {
  let summary = "Repeat tensor along each dimension";
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$repeats
  );
  let results = (outs AnyTensor:$output);
}

def Gawee_TopKOp : Gawee_Op<"topk", []> {
  let summary = "Top-K values and indices along one axis";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$k,
    I64Attr:$axis
  );
  // result 0 = values, result 1 = indices (unnamed to avoid OpAsmOpInterface trait)
  let results = (outs AnyTensor, AnyTensor);
}

def Gawee_CastOp : Gawee_Op<"cast", []> {
  let summary = "Tensor cast";
  let arguments = (ins
    AnyTensor:$input,
    I64Attr:$toType
  );
  let results = (outs AnyTensor:$output);

  // TODO when implementing:
  // - emitter: ONNX Cast uses a target dtype enum
  // - lowering: decide whether to lower with arith/fp/int cast ops directly
}

#endif
