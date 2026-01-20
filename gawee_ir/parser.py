# gawee_frontend/torch_parser.py

from __future__ import annotations
from typing import *

import torch
import torch.fx as fx
import numpy as np

from gawee_ir.graph import Graph, Node, Value


class TorchParser:

    @classmethod
    def parse_fx(
        cls,
        gm: fx.GraphModule,
        example_inputs: Tuple[torch.Tensor, ...],
    ) -> Graph:
        g = Graph()

        # --- 0) shape propagation ---
        from torch.fx.passes.shape_prop import ShapeProp
        ShapeProp(gm).propagate(*example_inputs)

        # --- 1) parameters / buffers -> constants ---
        params = dict(gm.named_parameters())
        buffers = dict(gm.named_buffers())

        for name, t in {**params, **buffers}.items():
            arr = t.detach().cpu().numpy()
            v = g.get_value(
                name=name,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
            )
            v.data = arr

        env: Dict[str, Value] = {}

        # --- 2) nodes ---
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                v = g.get_value(
                    name=node.name,
                    shape=list(node.meta["tensor_meta"].shape),
                    dtype=str(node.meta["tensor_meta"].dtype),
                )
                g.add_input(v)
                env[node.name] = v

            elif node.op == "get_attr":
                # parameter / buffer access
                v = g.get_value(name=node.target)
                env[node.name] = v

            elif node.op in ("call_function", "call_method", "call_module"):
                ins: List[Value] = []
                for arg in node.all_input_nodes:
                    ins.append(env[arg.name])

                # output
                tm = node.meta.get("tensor_meta", None)
                shape = list(tm.shape) if tm else None
                dtype = str(tm.dtype) if tm else None

                out = g.get_value(
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
                g.add_node(n)
                env[node.name] = out

            elif node.op == "output":
                # output can be tuple
                def extract(v):
                    if isinstance(v, fx.Node):
                        return env[v.name]
                    elif isinstance(v, (list, tuple)):
                        return [extract(x) for x in v]
                    else:
                        return None

                outs = extract(node.args[0])
                if isinstance(outs, list):
                    for v in outs:
                        g.add_output(v)
                else:
                    g.add_output(outs)

            else:
                raise NotImplementedError(f"Unsupported FX op: {node.op}")

        return g
