from __future__ import annotations
from typing     import *

import json
import os
import numpy as np
import torch

from gawee_ir.graph import Graph, Value, Node
from gawee_ir.extraction.attr_extractor import AttrExtractor

JSON_TYPE: TypeAlias = str | bool | None | int | float | List["JSON_TYPE"] | Dict[str, "JSON_TYPE"]

# Attributes that are weights/parameters and should be exported as binary.
WEIGHT_ATTRS = {"weight", "bias", "running_mean", "running_var"}


class Translator:

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.weight_dir = os.path.join(out_dir, "weights")
        os.makedirs(self.weight_dir, exist_ok=True)
        self._weight_counter = 0

    # ---------- helpers ----------

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def _export_tensor(self, arr: np.ndarray, name: str) -> Dict[str, JSON_TYPE]:
        """Export numpy array to binary file and return metadata."""
        fname = f"{name}.bin"
        path = os.path.join(self.weight_dir, fname)
        arr.tofile(path)

        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "path": f"weights/{fname}",
        }

    def _export_weight(self, v: Value) -> Dict[str, JSON_TYPE]:
        """Export constant Value to raw binary and return JSON metadata."""
        assert v.data is not None

        fname = f"{v.name}.bin"
        path = os.path.join(self.weight_dir, fname)

        arr = np.asarray(v.data)
        arr.tofile(path)

        return {
            "id": v.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "storage": {
                "format": "raw",
                "path": f"weights/{fname}",
            },
        }

    def _serialize_attr(self, key: str, val: Any, node_name: str) -> JSON_TYPE:
        """Convert a single attribute value to JSON-serializable form."""
        # Handle torch.Tensor -> export as binary
        if isinstance(val, torch.Tensor):
            arr = self._to_numpy(val)
            weight_name = f"{node_name}_{key}_{self._weight_counter}"
            self._weight_counter += 1
            return self._export_tensor(arr, weight_name)

        # Handle numpy array -> export as binary
        if isinstance(val, np.ndarray):
            weight_name = f"{node_name}_{key}_{self._weight_counter}"
            self._weight_counter += 1
            return self._export_tensor(val, weight_name)

        # Handle tuple -> convert to list
        if isinstance(val, tuple):
            return list(val)

        # Handle basic JSON types
        if isinstance(val, (str, bool, int, float, type(None))):
            return val

        # Handle list
        if isinstance(val, list):
            return [self._serialize_attr(key, item, node_name) for item in val]

        # Fallback: convert to string
        return str(val)

    # ---------- translation functions ----------

    def _value_to_json(self, v: Value) -> Dict[str, JSON_TYPE]:
        if v.is_const():
            return self._export_weight(v)

        return {
            "id": v.name,
            "shape": v.shape,  # type: ignore
            "dtype": v.dtype,
        }

    def _node_to_json(self, n: Node) -> Dict[str, Any]:
        # Re-extract attributes from raw fx.Node to get updated weights after passes.
        raw_attrs = AttrExtractor.extract(n.raw)

        # Serialize attributes, exporting tensors as binary files.
        serialized_attrs: Dict[str, JSON_TYPE] = {}
        for key, val in raw_attrs.items():
            if val is None and key in WEIGHT_ATTRS:
                # Skip None weights/biases (e.g., conv without bias)
                continue
            serialized_attrs[key] = self._serialize_attr(key, val, n.raw_name)

        return {
            "op_type": n.op_type,
            "name": n.raw_name,
            "inputs": [v.name for v in n.inputs],
            "outputs": [v.name for v in n.outputs],
            "attrs": serialized_attrs,
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
