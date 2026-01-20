# gawee_frontend/torch_parser.py

from __future__ import annotations
from typing import *

import torch
import torch.fx as fx
import numpy as np

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import *


class TorchParser:

    # preliminaries
    @classmethod
    def _extract(cls, v):
        if isinstance(v, fx.Node):
            return cls.env[v.name]
        elif isinstance(v, (list, tuple)):
            return [cls._extract(x) for x in v]
        else:
            return


    @classmethod 
    def _parse_placeholder(cls, node: fx.Node) -> None:
        v = cls.g.get_value(
            name=node.name,
            shape=list(node.meta[TENSOR_META].shape),
            dtype=str(node.meta[TENSOR_META].dtype),
        )
        cls.g.add_input(v)
        cls.env[node.name] = v
        return 


    @classmethod    
    def _parse_get_attr(cls, node: fx.Node) -> None:
        # parameter / buffer access
        v = cls.g.get_value(name=node.target)
        cls.env[node.name] = v
        return      

    
    @classmethod
    def _parse_call(cls, node: fx.Node) -> None:
        ins: List[Value] = []
        for arg in node.all_input_nodes:
            ins.append(cls.env[arg.name])

        # output
        tm = node.meta.get(TENSOR_META, None)
        shape = list(tm.shape) if tm else None
        dtype = str(tm.dtype) if tm else None

        out = cls.g.get_value(
            name=node.name,
            shape=shape,
            dtype=dtype,
        )

        attrs = {
            "target": str(node.target),
            "op": node.op,
        }

        n = Node(
            op_type=str(node.target),
            inputs=ins,
            outputs=[out],
            attrs=attrs,
            name=node.name,
        )
        cls.g.add_node(n)
        cls.env[node.name] = out
        return 

    
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
        cls.g = Graph()
        cls.env = dict()

        # --- 0) shape propagation ---
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*example_inputs)

        # --- 1) parameters / buffers -> constants ---
        params = dict(gm.named_parameters())
        buffers = dict(gm.named_buffers())

        for name, t in {**params, **buffers}.items():
            arr = t.detach().cpu().numpy()
            v = cls.g.get_value(
                name=name,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
            )
            v.data = arr

        # --- 2) nodes ---
        for node in gm.graph.nodes:
            print(f'operator -> {node.op}')
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
