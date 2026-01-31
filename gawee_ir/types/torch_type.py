from __future__ import annotations
from typing     import *

import torch 
import torch.nn as nn 

# torch module type. 
BN_TYPE: TypeAlias         = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
CONV_TYPE: TypeAlias       = nn.Conv1d | nn.Conv2d | nn.Conv3d
LINEAR_TYPE: TypeAlias     = nn.Linear
RELU_TYPE: TypeAlias       = nn.ReLU 
MXPOOL_TYPE: TypeAlias     = nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d
AVGPOOL_TYPE: TypeAlias    = nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d 
AD_AVGPOOL_TYPE: TypeAlias = nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d
ID_TYPE: TypeAlias         = nn.Identity