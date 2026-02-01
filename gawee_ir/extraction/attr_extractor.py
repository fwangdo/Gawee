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
        cls.attrs["op_type"] = mod.__class__.__name__
        cls.attrs["mod"] = mod  # Store module for passes

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
        elif isinstance(mod, AD_AVGPOOL_TYPE):
            return cls._extract_adaptive_avgpool(node, mod) 
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
        cls.attrs["in_features"] = mod.in_features
        cls.attrs["out_features"] = mod.out_features
        cls.attrs["weight"] = mod.weight
        cls.attrs["bias"] = mod.bias
        return 


    @classmethod
    def _extract_bn(cls, node: fx.Node, mod: BN_TYPE):
        cls.attrs["num_features"] = mod.num_features
        cls.attrs["weight"] = mod.weight # scale of bn. 
        cls.attrs["bias"] = mod.bias # bias of bn
        cls.attrs["affine"] = mod.affine # whether it's affine or not. 
        cls.attrs["running_mean"] = mod.running_mean # mu
        cls.attrs["running_var"] = mod.running_var # sigma ^ 2. 
        cls.attrs["num_batches_tracked"] = mod.num_batches_tracked
        cls.attrs["eps"] = mod.eps
        cls.attrs["momentum"] = mod.momentum
        cls.attrs["track_running_stats"] = mod.track_running_stats
        return 


    @classmethod
    def _extract_relu(cls, node: fx.Node, mod: RELU_TYPE):
        cls.attrs["inplace"] = mod.inplace
        return 

    
    @classmethod 
    def _extract_maxpool(cls, node: fx.Node, mod: MXPOOL_TYPE):
        cls.attrs["kernel_size"] = mod.kernel_size
        cls.attrs["stride"] = mod.stride
        cls.attrs["padding"] = mod.padding
        cls.attrs["dilation"] = mod.dilation
        cls.attrs["ceil_mode"] = mod.ceil_mode
        return 


    @classmethod
    def _extract_avgpool(cls, node: fx.Node, mod: AVGPOOL_TYPE):
        '''
        - kernel_size - pooling window size                                                                                                                                                                                                                                                                                  
        - stride - stride of the pooling window (defaults to kernel_size if None)                                                                                                                                                                                                                                            
        - padding - zero-padding added to both sides                                                                                                                                                                                                                                                                         
        - ceil_mode - whether to use ceil instead of floor for output shape computation                                                                                                                                                                                                                                      
        - count_include_pad - whether to include zero-padding in the averaging calculation   
        '''
        cls.attrs["kernel_size"] = mod.kernel_size
        cls.attrs["stride"] = mod.stride
        cls.attrs["padding"] = mod.padding
        cls.attrs["ceil_mode"] = mod.ceil_mode
        cls.attrs["count_include_pad"] = mod.count_include_pad
        return


    @classmethod 
    def _extract_adaptive_avgpool(cls, node: fx.Node, mod: AD_AVGPOOL_TYPE):
        cls.attrs["output_size"] = mod.output_size 
        return   


    @classmethod 
    def _extract_id(cls, node: fx.Node, mod: ID_TYPE):
        return 

    # handling for calls.
    @classmethod
    def _extract_call_method(cls, node: fx.Node):
        return node.target


    # changed by claude. 
    @classmethod
    def _extract_call_function(cls, node: fx.Node):
        """Extract attributes from call_function operations."""
        import torch
        import operator

        target = node.target

        # torch.flatten or Tensor.flatten
        if target in (torch.flatten,) or (hasattr(target, '__name__') and target.__name__ == 'flatten'): # type: ignore
            cls._extract_flatten_func(node)

        # torch.cat
        elif target == torch.cat:
            cls._extract_cat_func(node)

        # torch.add, operator.add
        elif target in (torch.add, operator.add):
            cls.attrs["op_type"] = ADD 

        # torch.mul, operator.mul
        elif target in (torch.mul, operator.mul):
            cls.attrs["op_type"] = MUL 

        # torch.sub, operator.sub
        elif target in (torch.sub, operator.sub):
            cls.attrs["op_type"] = SUB 

        # F.interpolate
        elif hasattr(target, '__name__') and target.__name__ == 'interpolate': # type: ignore
            cls._extract_interpolate_func(node)

        return target


    @classmethod
    def _extract_flatten_func(cls, node: fx.Node):
        """
        torch.flatten(input, start_dim=0, end_dim=-1)
        """
        cls.attrs["op_type"] = FLATTEN 

        # start_dim: args[1] or kwargs['start_dim'], default=0
        if len(node.args) > 1:
            cls.attrs["start_dim"] = node.args[1]
        else:
            cls.attrs["start_dim"] = 0 

        # end_dim: args[2] or kwargs['end_dim'], default=-1
        if len(node.args) > 2:
            cls.attrs["end_dim"] = node.args[2]
        else:
            cls.attrs["end_dim"] = -1 

        return


    @classmethod
    def _extract_cat_func(cls, node: fx.Node):
        """
        torch.cat(tensors, dim=0)
        """
        cls.attrs["op_type"] = CAT 

        # dim: args[1] or kwargs['dim'], default=0
        if len(node.args) > 1:
            cls.attrs["dim"] = node.args[1]
        else:
            cls.attrs["dim"] = 0 

        return


    @staticmethod
    def _sanitize_size(size) -> Tuple[int, ...] | None:
        """Convert size to tuple of ints. Return None if any dim is symbolic."""
        if size is None:
            return None
        if isinstance(size, int):
            return (size,)
        result = []
        for d in size:
            if isinstance(d, int):
                result.append(d)
            else:
                # Symbolic dimension (fx.Node, etc.) - size is dynamic
                return None
        return tuple(result)

    @classmethod
    def _extract_interpolate_func(cls, node: fx.Node):
        """
        F.interpolate(input, size=None, scale_factor=None, mode='nearest', ...)
        """
        cls.attrs["op_type"] = INTERPOLATE

        # size - may contain fx.Node for dynamic shapes, sanitize it
        raw_size = None
        if len(node.args) > 1:
            raw_size = node.args[1]
        else:
            raw_size = node.kwargs.get("size", None)
        cls.attrs["size"] = cls._sanitize_size(raw_size)

        # scale_factor
        if len(node.args) > 2:
            cls.attrs["scale_factor"] = node.args[2]
        else:
            cls.attrs["scale_factor"] = node.kwargs.get("scale_factor", None)

        # mode
        if len(node.args) > 3:
            cls.attrs["mode"] = node.args[3]
        else:
            cls.attrs["mode"] = node.kwargs.get("mode", "nearest")

        return


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

        print()
        print(f'node -> {node}')
        print(f'attrs -> {cls.attrs}')
        return cls.attrs 