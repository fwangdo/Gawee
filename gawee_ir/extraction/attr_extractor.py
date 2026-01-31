# To record all information from fx Node. 
from __future__ import annotations
from typing     import *

import torch.fx as fx
import numpy    as np

from gawee_ir.graph            import Graph, Node, Value
from gawee_ir.constant.ops     import *
from gawee_ir.mapper           import *
from gawee_ir.types.torch_type import *

import sys


class AttrExtractor:

    # preliminaries. 
    @classmethod 
    def _extract_call_module(cls, node: fx.Node):
        mod = cls.gm.get_submodule(node.target) # type: ignore 
        assert isinstance(mod, nn.Module), f'[ERROR]: mod -> {mod}'

        if isinstance(mod, CONV_TYPE):
            return cls._extract_conv(node, mod) 
        elif isinstance(mod, LINEAR_TYPE):
            return cls._extract_linear(node, mod) 
        elif isinstance(mod, BN_TYPE):
            return cls._extract_bn(node, mod) 
        elif isinstance(mod, RELU_TYPE):
            return cls._extract_relu(node, mod) 
        elif isinstance(mod, MXPOOL_TYPE):
            return cls._extract_maxpool(node, mod) 
        elif isinstance(mod, AVGPOOL_TYPE):
            return cls._extract_avgpool(node, mod) 
        elif isinstance(mod, ID_TYPE):
            return cls._extract_id(node, mod) 

        raise Exception(f'[ERROR]: {mod} is not defined yet in parse_module')


    @classmethod
    def _extract_conv(cls, node: fx.Node, mod: CONV_TYPE):
        '''
            in_channels (int) Number of channels in the input image
            out_channels (int) Number of channels produced by the convolution
            kernel_size (int or tuple) Size of the convolving kernel
            stride (int or tuple, optional) Stride of the convolution. Default: 1
            padding (int, tuple or str, optional) Padding added to both sides of the input. Default: 0
            dilation (int or tuple, optional) Spacing between kernel elements. Default: 1
            groups (int, optional) Number of blocked connections from input channels to output channels. Default: 1
            bias (bool, optional) If True, adds a learnable bias to the output. Default: True
            padding_mode (str, optional) 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        '''
        cls.attrs["out_channels"] = mod.out_channels
        cls.attrs["kernel_size"]  = mod.kernel_size
        cls.attrs["stride"]       = mod.stride 
        cls.attrs["padding"]      = mod.padding 
        cls.attrs["dilation"]     = mod.dilation 
        cls.attrs["groups"]       = mod.groups 
        cls.attrs["weight"]       = mod.weight
        cls.attrs["bias"]         = mod.bias
        return 


    @classmethod 
    def _extract_linear(cls, node: fx.Node, mod: LINEAR_TYPE):
        return 


    @classmethod
    def _extract_bn(cls, node: fx.Node, mod: BN_TYPE):
        return 


    @classmethod
    def _extract_relu(cls, node: fx.Node, mod: RELU_TYPE):
        return 

    
    @classmethod 
    def _extract_maxpool(cls, node: fx.Node, mod: MXPOOL_TYPE):
        return 


    @classmethod 
    def _extract_avgpool(cls, node: fx.Node, mod: AVGPOOL_TYPE):
        return   


    @classmethod 
    def _extract_id(cls, node: fx.Node, mod: ID_TYPE):
        return 
    # TODO 

    # handling for calls. 
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