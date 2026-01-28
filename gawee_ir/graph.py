# Definition of Gawee IR. 

from __future__            import annotations
from typing                import *  
import numpy               as np 
import torch.fx            as fx 
from gawee_ir.constant.ops import *

DimType = List[int]

class Value:

    def __init__(
        self,
        name: str,
        shape: DimType | None = None,
        dtype: str | None = None,
        data: np.ndarray | None = None,   
    ):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.data = data             

        self.producer: Node | None = None
        self.consumers: List[Node] = []

    def is_const(self) -> bool:
        return self.data is not None

    def __repr__(self) -> str:
        shape = self.shape if self.shape is not None else "?"
        dtype = self.dtype if self.dtype is not None else "?"
        const = " const" if self.is_const() else ""
        return f"Value(name={self.name}, shape={shape}, dtype={dtype}){const}"

    def to_json(self):
        # TODO 
        return 


class Node:

    def __init__(
        self,
        op_type: str,
        inputs: List[Value],
        outputs: List[Value],
        raw_name: str,  
        raw: fx.Node, # To get information. 

        attrs: Dict[str, Any] | None = None,
        name: str | None = None,
        call_type: str | None = None, 
    ):
        self.op_type = op_type # operator type.
        self.inputs = inputs
        self.outputs = outputs
        self.raw_name = raw_name
        self.raw = raw 

        self.attrs: Dict[str, Any] = attrs if attrs is not None else {}
        self.name = name
        self.call_type = call_type

        for v in self.inputs:
            v.consumers.append(self)
        for v in self.outputs:
            v.producer = self

    def __repr__(self) -> str:
        in_names = [v.name for v in self.inputs]
        out_names = [v.name for v in self.outputs]
        return (
            f"Node(op_type={self.op_type}, "
            f"inputs={in_names}, outputs={out_names}, "
            f"raw={in_names})"
        )

    def is_call_function(self) -> bool:
        return self.call_type == CALL_FUNCTION 

    def to_json(self):
        # TODO 
        return 


class Graph:
    def __init__(self):
        self.inputs: List[Value] = []
        self.outputs: List[Value] = []
        self.nodes: List[Node] = []
        self.values: Dict[str, Value] = {}

    # ---- Value helpers ----

    def get_value(
        self,
        name: Any,
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

        # create. 
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

    # ---- Rewrite utilities ----

    def make_const(
        self,
        data: np.ndarray,
        name: str | None = None,
        dtype: str | None = None,
    ) -> Value:
        """Create a constant Value registered in this graph."""
        if name is None:
            name = f"const_{len(self.values)}"

        base = name
        name = f"{base}_{len(self.values)}"

        v = Value(
            name=name,
            shape=list(data.shape),
            dtype=(dtype if dtype is not None else str(data.dtype).lower()),
            data=data,
        )
        self.values[v.name] = v
        return v


    def replace_all_uses(self, old: Value, new: Value) -> None:
        """Replace every use of `old` with `new` (inputs + graph outputs)."""
        if old is new:
            return

        # Update node inputs
        for user in list(old.consumers):
            replaced = False
            for i, inp in enumerate(user.inputs):
                if inp is old:
                    user.inputs[i] = new
                    replaced = True
            if replaced:
                if user not in new.consumers:
                    new.consumers.append(user)
                if user in old.consumers:
                    old.consumers.remove(user)

        # Update graph outputs
        for i, outv in enumerate(self.outputs):
            if outv is old:
                self.outputs[i] = new

        return

    def to_json(self):
        # TODO 
        return 

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

        print("\n[Values]")
        for v in self.values:
            print(f"  {v}")

        print("\n[Nodes]")
        for i, n in enumerate(self.nodes):
            print(f"  ({i}) {n}")

        print("\n[Outputs]")
        for v in self.outputs:
            print(f"  {v}")
        return 


    def show_node(self) -> None:
        print(f'\n\nCurrent Nodes.')
        for n in self.nodes:
            print(f'node -> {n.op_type}')
        return 