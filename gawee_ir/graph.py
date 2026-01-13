# gawee_ir/graph.py

from __future__ import annotations
from typing     import *  

DimType = List[int]

class Value:
    def __init__(
        self,
        name: str,
        shape: DimType | None = None,
        dtype: str | None = None,
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype

        self.producer: Node | None = None
        self.consumers: List[Node] = []

    def __repr__(self) -> str:
        shape = self.shape if self.shape is not None else "?"
        dtype = self.dtype if self.dtype is not None else "?"
        return f"Value(name={self.name}, shape={shape}, dtype={dtype})"


class Node:
    def __init__(
        self,
        op_type: str,
        inputs: List[Value],
        outputs: List[Value],
        attrs: Dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attrs: Dict[str, Any] = attrs if attrs is not None else {}
        self.name = name

        for v in self.inputs:
            v.consumers.append(self)
        for v in self.outputs:
            v.producer = self

    def __repr__(self) -> str:
        in_names = [v.name for v in self.inputs]
        out_names = [v.name for v in self.outputs]
        return (
            f"Node(op_type={self.op_type}, "
            f"inputs={in_names}, outputs={out_names})"
        )


class Graph:
    def __init__(self):
        self.inputs: List[Value] = []
        self.outputs: List[Value] = []
        self.nodes: List[Node] = []
        self.values: Dict[str, Value] = {}

    # ---- Value helpers ----

    def get_or_create_value(
        self,
        name: str,
        shape: DimType | None = None,
        dtype: str | None = None,
    ) -> Value:
        if name in self.values:
            v = self.values[name]
            if shape is not None:
                v.shape = shape
            if dtype is not None:
                v.dtype = dtype
            return v

        v = Value(name=name, shape=shape, dtype=dtype)
        self.values[name] = v
        return v

    # ---- Graph construction ----

    def add_input(self, value: Value) -> None:
        self.inputs.append(value)

    def add_output(self, value: Value) -> None:
        self.outputs.append(value)

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)

    # ---- Query utilities ----

    def find_nodes_by_op(self, op_type: str) -> List[Node]:
        return [n for n in self.nodes if n.op_type == op_type]

    def remove_node(self, node: Node) -> None:
        if node not in self.nodes:
            return

        for v in node.inputs:
            if node in v.consumers:
                v.consumers.remove(node)

        for v in node.outputs:
            v.producer = None

        self.nodes.remove(node)

    # ---- Debug / dump ----

    def dump(self) -> None:
        print("=== Graph ===")
        print("[Inputs]")
        for v in self.inputs:
            print(f"  {v}")

        print("\n[Nodes]")
        for i, n in enumerate(self.nodes):
            print(f"  ({i}) {n}")

        print("\n[Outputs]")
        for v in self.outputs:
            print(f"  {v}")
        return 
