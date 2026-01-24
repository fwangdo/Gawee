# TODO 
'''
- Redundant node eliminations
Identity Elimination
Slice Elimination
Unsqueeze Elimination
Dropout Elimination

- Opearator fusions. 
Conv Add Fusion
Conv Mul Fusion
Conv BatchNorm Fusion
Relu Clip Fusion
Reshape Fusion
'''
from __future__ import annotations
from typing     import *


class Passer:

    @classmethod
    def run(cls): 
        return 