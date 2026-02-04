#ifndef GAWEE_OPS_TD
#define GAWEE_OPS_TD

include "GaweeDialect.td"
include "mlir/IR/OpBase.td"

def Gawee_ConvOp : Gawee_Op<"conv", []> {
  let summary = "2D convolution";
  let arguments = (ins 
    AnyTensor:$input, 
    AnyTensor:$weight,
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

#endif
