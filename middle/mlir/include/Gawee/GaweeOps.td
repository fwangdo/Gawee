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

#endif
