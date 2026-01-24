from __future__ import annotations
from typing     import *

import torch 
import torch.nn as nn 

BN_TYPE = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
CONV_TYPE = nn.Conv1d | nn.Conv2d | nn.Conv3d