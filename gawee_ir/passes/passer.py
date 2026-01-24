'''
- Opearator fusions. 
Conv BatchNorm Fusion
Conv Add Fusion
- hereafter, these passes are not useful for resnet. 
Conv Mul Fusion
Relu Clip Fusion
Reshape Fusion

We do not consider this part now. Because these passes are not helpful to optimize resnet. 
- Redundant node eliminations
Identity Elimination
Slice Elimination
Unsqueeze Elimination
Dropout Elimination
'''
from __future__ import annotations
from typing     import *

from gawee_ir.passes.constant_folding import ConstantFolding 
from gawee_ir.passes.conv_bn_folding  import ConvBNFolding
from gawee_ir.passes.conv_add_folding import ConvAddFolding 
from gawee_ir.graph                   import Graph


class Passer:

    @classmethod
    def run(cls, g: Graph) -> Graph: 
        # fusions. 
        ConstantFolding.run(g)
        ConvBNFolding.run(g)
        ConvAddFolding.run(g)
        return g 