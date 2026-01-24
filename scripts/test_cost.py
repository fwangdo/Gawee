from gawee_ir.parser import TorchParser
from gawee_ir.analysis.shape import ShapeInference
from gawee_ir.analysis.cost import CostModel

# passes. 
from gawee_ir.passes.conv_bn_folding import ConvBNFolding
from gawee_ir.passes.constant_folding import *

# preliminaries. 
import torch    
from torchvision.models import resnet18
import torch.fx as fx
import numpy as np

path = './torchdata/resnet18.pt'

model = resnet18(weights=None)
state_dict = torch.load(path)
model.load_state_dict(state_dict)
model.eval()

gm = fx.symbolic_trace(model)
g = TorchParser.parse_fx(gm, (torch.randn(1, 3, 224, 224),))    

ShapeInference.run(g)

print("== Before ==")
CostModel.init(gm)
CostModel.print_report(g)

# ConvBNFolding.run(g)
ConstantFolding.run(g)
ConvBNFolding.run(g)

print("\n\n== After ==")
CostModel.print_report(g)
