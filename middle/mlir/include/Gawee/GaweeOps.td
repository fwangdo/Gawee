//===----------------------------------------------------------------------===//
// Gawee Operations Definition
//===----------------------------------------------------------------------===//
//
// LEARNING: Defining Operations
//
// Each op needs:
//   1. Mnemonic: name in IR (e.g., "conv" -> gawee.conv)
//   2. Arguments: inputs (operands) and attributes
//   3. Results: output types
//   4. Traits: properties (Pure, SameOperandsAndResultType, etc.)
//
// Key concepts:
//   - Operand: runtime value (tensor, memref)
//   - Attribute: compile-time constant (int, array)
//   - Trait: semantic property the op guarantees
//
//===----------------------------------------------------------------------===//

#ifndef GAWEE_OPS_TD
#define GAWEE_OPS_TD

include "GaweeDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Conv2D Operation
//===----------------------------------------------------------------------===//
//
// LEARNING: This is the most complex op. Study it carefully.
//
// In Python you had:
//   attrs = {kernel_size, stride, padding, ...}
//   inputs = [activation, weight, bias]
//   outputs = [result]
//
// In MLIR, same info but declarative:
//

def Gawee_ConvOp : Gawee_Op<"conv", [Pure]> {
  let summary = "2D convolution operation";

  let description = [{
    Performs 2D convolution on input tensor with kernel weights.

    Example:
    ```mlir
    %out = gawee.conv %in, %weight, %bias {
      strides = [1, 1],
      padding = [0, 0],
      dilation = [1, 1]
    } : tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>
      -> tensor<1x64x222x222xf32>
    ```
  }];

  // Operands: runtime tensor values
  // TODO: Define operands
  // Hint: Use AnyTensor for input, weight, bias
  //   let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias, ...);
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    // TODO: bias is optional (some convs have no bias)
    // Hint: Use Optional<AnyTensor>:$bias

    // Attributes: compile-time constants
    // TODO: Add stride, padding, dilation attributes
    // Hint: Use DenseI64ArrayAttr for arrays
    //   DenseI64ArrayAttr:$strides,
    //   DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding,
    DenseI64ArrayAttr:$dilation
  );

  // Results: output tensors
  let results = (outs AnyTensor:$output);

  // TODO: Add verifier to check input/output shape consistency
  // let hasVerifier = 1;

  // TODO: Add custom builder for convenient C++ construction
  // let builders = [...];
}

//===----------------------------------------------------------------------===//
// ReLU Operation
//===----------------------------------------------------------------------===//
//
// LEARNING: Simple elementwise op
//
// Traits:
//   - Pure: no side effects, same input -> same output
//   - SameOperandsAndResultType: input and output have same type
//

def Gawee_ReluOp : Gawee_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";

  let description = [{
    Applies ReLU (Rectified Linear Unit): out = max(0, in)

    Example:
    ```mlir
    %out = gawee.relu %in : tensor<1x64x112x112xf32>
    ```
  }];

  // TODO: Define arguments and results
  // Hint: Single tensor input, single tensor output
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  // LEARNING: Assemby format defines how the op is printed/parsed
  // This makes: gawee.relu %x : tensor<...>
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// MaxPool Operation
//===----------------------------------------------------------------------===//

def Gawee_MaxPoolOp : Gawee_Op<"maxpool", [Pure]> {
  let summary = "2D max pooling operation";

  let description = [{
    Performs max pooling over spatial dimensions.

    Example:
    ```mlir
    %out = gawee.maxpool %in {
      kernel_size = [2, 2],
      strides = [2, 2]
    } : tensor<1x64x112x112xf32> -> tensor<1x64x56x56xf32>
    ```
  }];

  // TODO: Define arguments
  // Hint: input tensor + kernel_size, strides, padding attributes
  let arguments = (ins
    AnyTensor:$input,
    DenseI64ArrayAttr:$kernel_size,
    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$padding
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// Add Operation (elementwise)
//===----------------------------------------------------------------------===//

def Gawee_AddOp : Gawee_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise addition";

  // TODO: Define for two input tensors
  // Hint: Similar structure to ReLU but with two inputs
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
}

//===----------------------------------------------------------------------===//
// Linear (Fully Connected) Operation
//===----------------------------------------------------------------------===//

def Gawee_LinearOp : Gawee_Op<"linear", [Pure]> {
  let summary = "Fully connected / linear layer";

  let description = [{
    Computes: output = input @ weight^T + bias

    Example:
    ```mlir
    %out = gawee.linear %in, %weight, %bias
      : tensor<1x512xf32>, tensor<1000x512xf32>, tensor<1000xf32>
      -> tensor<1x1000xf32>
    ```
  }];

  // TODO: Define arguments (input, weight, optional bias)
  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias
  );

  let results = (outs AnyTensor:$output);
}

//===----------------------------------------------------------------------===//
// TODO: Add more operations as needed
//===----------------------------------------------------------------------===//
//
// Operations to consider:
//   - Gawee_BatchNormOp (but you fold it in Python, so maybe not needed?)
//   - Gawee_ConcatOp (for skip connections)
//   - Gawee_AdaptiveAvgPoolOp
//   - Gawee_FlattenOp
//
// LEARNING: Start minimal, add ops as you need them for lowering.
//

#endif // GAWEE_OPS_TD
