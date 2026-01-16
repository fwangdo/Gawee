from gawee_ir.parser import Parser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.analysis.cost import CostModel

g = Parser.parse_onnx("./onnxdata/resnet18.onnx")
ShapeInference.run(g)
CostModel.print_report(g, topk=30)