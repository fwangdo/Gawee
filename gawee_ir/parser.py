# gawee_frontend/torch_parser.py

from __future__ import annotations
from typing import *

import torch
import torch.fx as fx
import numpy as np

from gawee_ir.graph import Graph, Node, Value, DimType
from gawee_ir.constant.ops import *
from gawee_ir.mapper import *
from gawee_ir.extraction.attr_extractor import *
from gawee_ir.analysis.shape import TorchShapeAnalyzer


class TorchParser:

    @staticmethod
    def _to_shape(shape: torch.Size) -> List[int]:
        """Convert torch.Size to List[int]. Fails on symbolic dims."""
        result = []
        for d in shape:
            if isinstance(d, int):
                result.append(d)
            elif isinstance(d, fx.Node):
                # TODO. 
                raise TypeError(f"Symbolic dimension not supported: {d}")
            else:
                raise TypeError(f"Unknown dimension type: {type(d)}")
        return result


    @staticmethod
    def _to_dtype(dtype: torch.dtype) -> str:
        """Convert torch.dtype to string."""
        return str(dtype)


    @classmethod
    def _get_shape(cls, node: fx.Node) -> DimType:
        tm = node.meta.get(TENSOR_META)
        if tm is not None:
            shape = cls._to_shape(tm.shape)
        else:
            shape = TorchShapeAnalyzer.infer_shape(node)
        
        if shape is not None:
            cls.shape[node.name] = shape
        return shape 

        
    @classmethod
    def _get_dtype(cls, node: fx.Node) -> str:
        tm = node.meta.get(TENSOR_META)
        if tm is not None:
            dtype = cls._to_dtype(tm.dtype)
        else:
            dtype = TorchShapeAnalyzer.infer_dtype(node) 

        if dtype is not None:
            cls.dtype[node.name] = dtype
        return dtype 


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
    def _parse_call_name(cls, node: fx.Node):
        if node.op == CALL_FUNCTION:
            return cls._parse_call_function(node)
        elif node.op == CALL_METHOD:
            return cls._parse_call_method(node)
        elif node.op == CALL_MODULE:
            return cls._parse_call_module(node)
        else:
            raise Exception(f'{node.op} is not supported yet')


    @classmethod
    def _extract(cls, v):
        if isinstance(v, fx.Node):
            return cls.env[v.name]
        elif isinstance(v, (list, tuple)):
            return [cls._extract(x) for x in v]
        else:
            raise Exception(f'[ERROR]: v -> {v} in {type(v)}')


    # parsing for each operation type.
    @classmethod
    def _parse_placeholder(cls, node: fx.Node) -> None:
        tm = node.meta[TENSOR_META]
        v = cls.g.get_value(
            name=node.name,
            shape=cls._to_shape(tm.shape),
            dtype=cls._to_dtype(tm.dtype),
        )
        cls.g.add_input(v)
        cls.env[node.name] = v
        return 


    @classmethod
    def _parse_get_attr(cls, node: fx.Node) -> None:
        v = cls.g.get_value(name=node.target)
        cls.env[node.name] = v
        return 


    @classmethod
    def _parse_call(cls, node: fx.Node) -> None:
        ins: List[Value] = []
        for arg in node.all_input_nodes:
            ins.append(cls.env[arg.name])

        # output shape/dtype from PyTorch's ShapeProp
        shape = cls._get_shape(node)
        dtype = cls._get_dtype(node)

        out = cls.g.get_value(
            name=node.name,
            shape=shape,
            dtype=dtype,
        )

        op_type = Mapper.translate(node, cls.gm)
        attrs = AttrExtractor.extract(node)

        n = Node(
            op_type=op_type,
            inputs=ins,
            outputs=[out],
            raw_name=str(node.target),
            raw=node,
            attrs=attrs,
            name=node.name,
            call_type=node.op,
        )
        cls.g.add_node(n)
        cls.env[node.name] = out


    @classmethod
    def _parse_output(cls, node: fx.Node) -> None:
        outs = cls._extract(node.args[0])
        if isinstance(outs, list):
            for v in outs:
                cls.g.add_output(v)
        else:
            cls.g.add_output(outs)
        return 


    @classmethod
    def parse_fx(
        cls,
        gm: fx.GraphModule,
        example_inputs: Tuple[torch.Tensor, ...],
    ) -> Graph:
        cls.gm = gm
        AttrExtractor.init(gm)

        cls.g = Graph()
        cls.env = dict()

        # to handle symbolic shapes. 
        cls.shape = dict()
        cls.dtype = dict()
        TorchShapeAnalyzer.init(gm, cls.shape, cls.dtype)

        # shape propagation with concrete inputs
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*example_inputs)

        # parameters / buffers -> constants
        params = dict(gm.named_parameters())
        buffers = dict(gm.named_buffers())

        for name, t in {**params, **buffers}.items():
            arr: np.ndarray = t.detach().cpu().numpy()
            v = cls.g.get_value(
                name=name,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
            )
            v.data = arr

        # nodes
        for node in gm.graph.nodes:
            if node.op == PLACEHOLDER:
                cls._parse_placeholder(node)
            elif node.op == GET_ATTR:
                cls._parse_get_attr(node)
            elif node.op in { CALL_FUNCTION, CALL_METHOD, CALL_MODULE }:
                cls._parse_call(node)
            elif node.op == OUTPUT:
                cls._parse_output(node)
            else:
                raise NotImplementedError(f"Unsupported FX op: {node.op}")

        return cls.g
