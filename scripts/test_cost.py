from gawee_ir.parser import Parser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.analysis.cost import CostModel
from gawee_ir.passes.conv_bn_folding import ConvBNFolding

g = Parser.parse_onnx("./onnxdata/resnet18.onnx")
ShapeInference.run(g)

print("== Before ==")
CostModel.print_report(g)

ConvBNFolding.run(g)

print("== After ==")
CostModel.print_report(g)
