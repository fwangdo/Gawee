from __future__ import annotations
from typing     import *

# onnx 
import onnx
import onnx.numpy_helper as numpy_helper

# IR 
from gawee_ir.graph import Graph, Node, Value


class Parser: 

    @classmethod
    def _get_tensor_shape(cls, value_info: onnx.ValueInfoProto) -> List[int] | None:
        if not value_info.type.HasField("tensor_type"):
            return None
        shape = []
        for d in value_info.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value))
            else:
                # unknown / symbolic dim
                shape.append(-1)
        return shape


    @classmethod
    def parse_onnx(cls, path: str) -> Graph:
        model = onnx.load(path)
        gp = model.graph
        g = Graph()

        # --- 1) initializers: weight constants ---
        for init in gp.initializer:
            arr = numpy_helper.to_array(init)  # np.ndarray
            v = g.get_value(
                name=init.name,
                shape=list(arr.shape),
                dtype=str(arr.dtype).lower(),
            )
            v.data = arr

        input_names = set(init.name for init in gp.initializer)

        # --- 2) graph inputs (exclude initializers) ---
        for inp in gp.input:
            if inp.name in input_names:
                continue
            shape = cls._get_tensor_shape(inp)
            # elem_type -> 아직 string으로 박아둔 상태면, 나중에 mapper를 두는 게 좋습니다.
            dtype = str(inp.type.tensor_type.elem_type)
            v = g.get_value(name=inp.name, shape=shape, dtype=dtype)
            g.add_input(v)

        # --- 3) nodes ---
        for node_proto in gp.node:
            ins = [g.get_value(name=n) for n in node_proto.input]
            outs = [g.get_value(name=n) for n in node_proto.output]

            attrs = {}
            for a in node_proto.attribute:
                # 최소 구현: ints/floats/i/s
                if a.type == onnx.AttributeProto.INTS:
                    attrs[a.name] = list(a.ints)
                elif a.type == onnx.AttributeProto.FLOATS:
                    attrs[a.name] = list(a.floats)
                elif a.type == onnx.AttributeProto.INT:
                    attrs[a.name] = int(a.i)
                elif a.type == onnx.AttributeProto.FLOAT:
                    attrs[a.name] = float(a.f)
                elif a.type == onnx.AttributeProto.STRING:
                    attrs[a.name] = a.s.decode("utf-8", errors="ignore")

            n = Node(
                op_type=node_proto.op_type,
                inputs=ins,
                outputs=outs,
                attrs=attrs,
                name=node_proto.name if node_proto.name else None,
            )
            g.add_node(n)

        # --- 4) outputs ---
        for out in gp.output:
            v = g.get_value(name=out.name)
            g.add_output(v)

        return g