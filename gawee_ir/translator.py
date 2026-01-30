from __future__ import annotations
from typing     import * 

import json
import os
import numpy as np

from gawee_ir.graph import Graph, Value, Node

BASIC_TYPE = str | bool | None | int | float 
HIGHER_TYPE = List[BASIC_TYPE] | Dict[str, BASIC_TYPE]
JSON_TYPE = BASIC_TYPE | HIGHER_TYPE

class Translator:

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.weight_dir = os.path.join(out_dir, "weights")
        os.makedirs(self.weight_dir, exist_ok=True)

    # ---------- helpers ----------

    def _export_weight(self, v: Value) -> Dict[str, Any]:
        """
        Export constant Value to raw binary and return JSON metadata.
        """
        assert v.data is not None

        fname = f"{v.name}.bin"
        path = os.path.join(self.weight_dir, fname)

        arr = np.asarray(v.data)
        arr.astype(arr.dtype).tofile(path)

        return {
            "id": v.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "storage": {
                "format": "raw",
                "path": f"weights/{fname}",
            },
        }

    # translation function. 

    def _value_to_json(self, v: Value) -> Dict[str, JSON_TYPE]:
        if v.is_const():
            return self._export_weight(v)

        return {
            "id": v.name,
            "shape": v.shape, # type: ignore
            "dtype": v.dtype,
        }

    def _node_to_json(self, n: Node) -> Dict[str, JSON_TYPE]:
        return {
            "op_type": n.op_type,
            "inputs": [v.name for v in n.inputs],
            "outputs": [v.name for v in n.outputs],
            "attrs": n.attrs,
        }

    # ---------- public ----------

    def export(self, graph: Graph, fname: str = "graph.json") -> None:
        graph_json = {
            "inputs": [v.name for v in graph.inputs],
            "outputs": [v.name for v in graph.outputs],
            "values": {
                name: self._value_to_json(v) for name, v in graph.values.items()
            },
            "nodes": [self._node_to_json(n) for n in graph.nodes],
        }

        out_path = os.path.join(self.out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(graph_json, f, indent=2)

        print(f"[LOG]: exported graph → {out_path}")
        return 