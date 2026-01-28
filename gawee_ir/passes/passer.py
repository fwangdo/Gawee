'''
- Opearator fusions. 
Conv BatchNorm Fusion(done)
Conv Add Fusion(done)
- hereafter, these passes are not useful for resnet. 
Conv Mul Fusion
Relu Clip Fusion
Reshape Fusion

We do not consider this part now. Because these passes are not helpful to optimize resnet. 
- Redundant node eliminations
Identity Elimination(done)
Slice Elimination
Unsqueeze Elimination
Dropout Elimination
'''
from __future__ import annotations
from typing     import *

# passes. 
from gawee_ir.passes.constant_folding import ConstantFolding 
from gawee_ir.passes.conv_bn_folding  import ConvBNFolding
from gawee_ir.passes.conv_add_folding import ConvAddFolding 
from gawee_ir.passes.elim_identity    import IdentityElimination
from gawee_ir.passes.canonicalize     import PythonOpElimination

# ir. 
from gawee_ir.graph                   import Graph


class Passer:

    result: Dict[str, int] = dict()

    @classmethod 
    def _record_pass_result(cls) -> None:
        cls.result[IdentityElimination.__name__] = IdentityElimination.deleted_node
        cls.result[ConvBNFolding.__name__] = ConvBNFolding.deleted_node
        cls.result[ConvAddFolding.__name__] = ConvAddFolding.deleted_node
        cls.result[PythonOpElimination.__name__] = PythonOpElimination.deleted_node
        return 


    @classmethod 
    def show_opt_result(cls) -> None:
        for opt, nodes in cls.result.items():
            print(f'{opt} -> {nodes}')
        return 


    @classmethod
    def run(cls, g: Graph) -> Graph: 
        # canonicalization. Delete python operations which can be represented as a graph.(e.g., getitem, getattr.)
        PythonOpElimination.run(g)

        # elimination. 
        IdentityElimination.run(g) 

        # fusions. 
        ConstantFolding.run(g)
        ConvBNFolding.run(g)
        ConvAddFolding.run(g)

        cls._record_pass_result()
        return g 


# note that, we can eliminate python operations(e.g., getattr, getitem). Consider how to delete them. 