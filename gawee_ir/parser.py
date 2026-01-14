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
        onnx.checker.check_model(model)
        graph_proto = model.graph

        g = Graph()

        # ------------------------------------------------------------
        # 1) Graph inputs
        # ------------------------------------------------------------
        input_names = set(init.name for init in graph_proto.initializer)

        for inp in graph_proto.input:
            if inp.name in input_names:
                # constant cases. 
                continue  

            shape = cls._get_tensor_shape(inp)
            v = g.get_value(
                name=inp.name,
                shape=shape,
                dtype=str(inp.type.tensor_type.elem_type),
            )
            g.add_input(v)

        # ------------------------------------------------------------
        # 2) Initializers (constants / weights)
        # ------------------------------------------------------------
        for init in graph_proto.initializer:
            arr = numpy_helper.to_array(init)
            g.get_value(
                name=init.name,
                shape=list(arr.shape),
                dtype=str(arr.dtype),
            )

        # ------------------------------------------------------------
        # 3) Nodes
        # ------------------------------------------------------------
        for node_proto in graph_proto.node:
            inputs: List[Value] = []
            outputs: List[Value] = []

            for name in node_proto.input:
                if name == "":
                    continue
                inputs.append(g.get_value(name))

            for name in node_proto.output:
                outputs.append(g.get_value(name))

            attrs: Dict[str, Any] = {}
            for attr in node_proto.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    attrs[attr.name] = attr.i
                elif attr.type == onnx.AttributeProto.FLOAT:
                    attrs[attr.name] = attr.f
                elif attr.type == onnx.AttributeProto.INTS:
                    attrs[attr.name] = list(attr.ints)
                elif attr.type == onnx.AttributeProto.FLOATS:
                    attrs[attr.name] = list(attr.floats)
                elif attr.type == onnx.AttributeProto.STRING:
                    attrs[attr.name] = attr.s.decode("utf-8")
                else:
                    # Tensor / Graph / SparseTensor 등은 일단 무시
                    pass

            n = Node(
                op_type=node_proto.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=attrs,
                name=node_proto.name or None,
            )
            g.add_node(n)

        # ------------------------------------------------------------
        # 4) Graph outputs
        # ------------------------------------------------------------
        for out in graph_proto.output:
            v = g.get_value(out.name)
            g.add_output(v)

        return g
