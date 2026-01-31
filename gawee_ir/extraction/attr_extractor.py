# To record all information from fx Node. 
from __future__ import annotations
from typing     import *

import torch.fx as fx
import numpy    as np

from gawee_ir.graph        import Graph, Node, Value
from gawee_ir.constant.ops import *
from gawee_ir.mapper       import *

import sys


class AttrExtractor:

    # preliminaries. 
    @classmethod 
    def _extract_call_module(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        assert isinstance(mod, nn.Module), f'[ERROR]: mod -> {mod}'

        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return cls._extract_conv(node) 
        elif isinstance(mod, nn.Linear):
            return cls._extract_linear(node) 
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return cls._extract_bn(node) 
        elif isinstance(mod, nn.ReLU):
            return cls._extract_bn(node) 
        elif isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            return cls._extract_maxpool(node) 
        elif isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d
                              , nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            return cls._extract_avgpool(node) 
        elif isinstance(mod, (nn.Identity)):
            return cls._extract_id(node) 

        raise Exception(f'[ERROR]: {mod} is not defined yet in parse_module')


    @classmethod
    def _extract_conv(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        print(f'mod -> {mod}')
        print(f'dir -> {dir(mod)}')
        sys.exit(1)
        return 


    @classmethod 
    def _extract_linear(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return 


    @classmethod
    def _extract_bn(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return 


    @classmethod
    def _extract_relu(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return 

    
    @classmethod 
    def _extract_maxpool(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return 


    @classmethod 
    def _extract_avgpool(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return   


    @classmethod 
    def _extract_id(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        return 
    # TODO 

    @classmethod 
    def _extract_call_method(cls, node: fx.Node):
        return node.target 


    @classmethod 
    def _extract_call_function(cls, node: fx.Node):
        return node.target 


    @classmethod 
    def _extract_call(cls, node: fx.Node): 
        if node.op == CALL_FUNCTION: 
            return cls._extract_call_function(node) 
        elif node.op == CALL_METHOD:
            return cls._extract_call_method(node) 
        elif node.op == CALL_MODULE:
            return cls._extract_call_module(node) 
        else:
            raise Exception(f'{node.op} is not supported yet')

    # main.  
    @classmethod
    def init(cls, gm: fx.GraphModule) -> None:
        cls.gm = gm
        return 

    @classmethod
    def extract(cls, node: fx.Node) -> Dict[str, Any]:
        cls.attrs: Dict[str, Any] = {
            "target": node.target, # dl opearation. 
            "op": node.op,
        }
        cls._extract_call(node) 
        return cls.attrs 