from __future__ import annotations
from typing     import *

import torch 
import torch.nn as nn 

# torch module type. 
BN_TYPE         = nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d
CONV_TYPE       = nn.Conv1d | nn.Conv2d | nn.Conv3d
LINEAR_TYPE     = nn.Linear
RELU_TYPE       = nn.ReLU 
MXPOOL_TYPE     = nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d
AVGPOOL_TYPE    = nn.AvgPool1d | nn.AvgPool2d | nn.AvgPool3d 
AD_AVGPOOL_TYPE = nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d
ID_TYPE         = nn.Identity