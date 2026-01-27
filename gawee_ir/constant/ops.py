# operators. 
RELU       = "Relu"
SIGMOID    = "Sigmoid"
TANH       = "Tanh"
ID         = "Identity"
BATCH_NORM = "BatchNormalization"
ADD        = "Add"
MUL        = "Mul"
SUB        = "Sub"
DIV        = "Div"
CONV       = "Conv"
MATMUL     = "MatMul"
GEMM       = "Gemm"
RESHAPE    = "Reshape"
TRANS      = "Transpose"

# extra ops used by some ONNX graphs
MAXPOOL    = "MaxPool"
AVGPOOL    = "AvgPool"
REDUCE_MEAN = "ReduceMean"
IDENTITY    = "Identity"

# FX layouts. 
PLACEHOLDER = "placeholder"
GET_ATTR = "get_attr"
CALL_FUNCTION = "call_function"
CALL_METHOD = "call_method"
CALL_MODULE = "call_module"
TENSOR_META = "tensor_meta"
OUTPUT = "output"

# operators. 
FLATTEN = "flatten"

# operators for call function 
GETATTR = "getattr"
GETITEM = "getitem"
INTERPOLATE = "interpolate"
CAT         = "cat"

SHAPE = "shape"
DTYPE = "dtype"