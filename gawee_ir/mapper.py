from __future__ import annotations
from typing     import * 

import torch.fx as fx
from gawee_ir.constant.ops import *
import torch.nn as nn

class Mapper:

    @classmethod
    def _extract_axes(cls, node: fx.Node):
        print(f'operator -> {node}')

        # dim can be in args or kwargs
        if "dim" in node.kwargs:
            dim = node.kwargs["dim"]
        elif len(node.args) >= 2:
            dim = node.args[1]
        else:
            return

        # normalize: int -> tuple[int]
        if isinstance(dim, int):
            return (dim,)
        if isinstance(dim, (list, tuple)):
            return tuple(dim)

        return 


    @classmethod
    def _extract_keepdims(cls, node: fx.Node):
        if "keepdim" in node.kwargs:
            return int(bool(node.kwargs["keepdim"]))
        return 


    # parsing for each call.  
    @classmethod 
    def _parse_call_module(cls, node: fx.Node):
        return cls.gm.get_submodule(node.target) # type: ignore 


    @classmethod 
    def _parse_call_method(cls, node: fx.Node):
        return node.target 


    @classmethod 
    def _parse_call_function(cls, node: fx.Node):
        return node.target 


    @classmethod 
    def _parse_primitive(cls, mod, node: fx.Node) -> str:
        """
        Extract the name of function used in CALL_FUNCTION in fx.

        Args:
            mod: module wrapped in torch.
            node: fx Node for debugging
        Return:
            the name of function 
        """
        mod_name = mod.__name__

        if mod_name == "add":
            return ADD  
        elif mod_name == "sub": 
            return SUB
        elif mod_name == "mul":
            return MUL
        elif mod_name == "flatten":
            return FLATTEN
        
        # added for unet. TODO: we need to consider parts below.  
        elif mod_name == "getattr":
            return GETATTR
        elif mod_name == "getitem": 
            return GETITEM
        elif mod_name == "interpolate":
            return INTERPOLATE
        elif mod_name == "cat":
            return CAT

        raise Exception(f'[ERROR] {mod} is not supported, name -> {mod_name}')


    @classmethod
    def _parse_module_name(cls, mod: nn.Module) -> str:  
        assert isinstance(mod, nn.Module), f'[ERROR]: mod -> {mod}'
        # print(f'mod -> {mod}, {type(mod)}')

        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            return CONV
        if isinstance(mod, nn.Linear):
            return MATMUL
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return BATCH_NORM
        if isinstance(mod, nn.ReLU):
            return RELU
        if isinstance(mod, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
            return MAXPOOL
        if isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
            return AVGPOOL
        if isinstance(mod, (nn.Identity)):
            return IDENTITY 

        raise Exception(f'[ERROR]: {mod} is not defined yet in parse_module')


    @classmethod 
    def _refine_op(cls, op: str, n, node: fx.Node) -> str: 
        if op == CALL_MODULE:
            res = cls._parse_module_name(n) 
        elif op == CALL_FUNCTION: 
            res = cls._parse_primitive(n, node) # node for debugging. 
        elif op == CALL_METHOD:
            raise Exception(f'[ERROR]: {n} is called by call method')
        else:
            raise Exception(f'[ERROR]: {n.op_type} is not supported yet. ')
        
        # print(f'refined operator -> {res}')
        return res 

    
    @classmethod 
    def translate(cls, node: fx.Node, gm: fx.GraphModule) -> str: 
        cls.gm = gm

        if node.op == CALL_FUNCTION: 
            temp = cls._parse_call_function(node) 
        elif node.op == CALL_METHOD:
            temp = cls._parse_call_method(node) 
        elif node.op == CALL_MODULE:
            temp = cls._parse_call_module(node) 
        else:
            raise Exception(f'{node.op} is not supported yet')

        res = cls._refine_op(node.op, temp, node)
        return res 


    # @classmethod
    # def translate(cls, node: fx.Node): 
    #     return 