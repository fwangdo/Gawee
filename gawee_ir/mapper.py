from __future__ import annotations
from typing     import * 

import torch.fx as fx
import torch.nn as nn

# type and constants. 
from gawee_ir.constant.ops     import *
from gawee_ir.types.torch_type import *

class Mapper:

    @classmethod
    def _extract_axes(cls, node: fx.Node):
        # print(f'operator -> {node}')

        # dim can be in args or kwargs
        if "dim" in node.kwargs:
            dim = node.kwargs["dim"]
        elif len(node.args) >= 2:
            dim = node.args[1]
        else:
            return

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
        
        # added for unet. 
        elif mod_name == "getattr":
            return GETATTR
        elif mod_name == "getitem": 
            return GETITEM
        elif mod_name == "interpolate":
            return INTERPOLATE
        elif mod_name == "cat": # concatenate
            return CAT

        raise Exception(f'[ERROR] {mod} is not supported, name -> {mod_name}')


    @classmethod
    def _parse_module_name(cls, mod: nn.Module) -> str:  
        assert isinstance(mod, nn.Module), f'[ERROR]: mod -> {mod}'
        # print(f'mod -> {mod}, {type(mod)}')

        # if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if isinstance(mod, CONV_TYPE):
            return CONV
        elif isinstance(mod, LINEAR_TYPE):
            return MATMUL
        elif isinstance(mod, BN_TYPE):
            return BATCH_NORM
        elif isinstance(mod, RELU_TYPE):
            return RELU
        elif isinstance(mod, MXPOOL_TYPE):
            return MAXPOOL
        elif isinstance(mod, AVGPOOL_TYPE):
            return AVGPOOL
        elif isinstance(mod, AD_AVGPOOL_TYPE):
            return AD_AVGPOOL
        elif isinstance(mod, ID_TYPE):
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

    
    # Hereby, we will extract features from fx Node to generate json file.  