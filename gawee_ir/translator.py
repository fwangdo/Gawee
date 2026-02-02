from __future__ import annotations
from typing     import *

import json
import os
import numpy as np
import torch

from gawee_ir.graph import Graph, Value, Node
from gawee_ir.extraction.attr_extractor import AttrExtractor
from gawee_ir.constant.ops import *

JSON_TYPE: TypeAlias = str | bool | None | int | float | List["JSON_TYPE"] | Dict[str, "JSON_TYPE"]

# Attributes that are weights/parameters and should be exported as binary.
# WEIGHT_ATTRS = {"weight", "bias", "running_mean", "running_var"}
WEIGHT_ATTRS = {WEIGHT, BIAS, RUNNING_MEAN, RUNNING_VAR}


class Translator:

    def __init__(self, out_dir: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.weight_dir = os.path.join(out_dir, "weights")
        self.const_dir = os.path.join(out_dir, "constants")
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.const_dir, exist_ok=True)
        self._weight_counter = 0

    # ---------- helpers ----------

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def _export_tensor(self, arr: np.ndarray, name: str) -> Dict[str, JSON_TYPE]:
        """Export weight tensor (from Node.attrs) to weights/ directory."""
        fname = f"{name}.bin"
        path = os.path.join(self.weight_dir, fname)
        arr.tofile(path)

        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "path": f"weights/{fname}",
        }

    def _export_constant(self, v: Value) -> Dict[str, JSON_TYPE]:
        """Export constant Value to constants/ directory."""
        assert v.data is not None

        fname = f"{v.name}.bin"
        path = os.path.join(self.const_dir, fname)

        arr = np.asarray(v.data)
        arr.tofile(path)

        return {
            "id": v.name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "path": f"constants/{fname}",
        }

    # serialization. 
    def _check_is_need_attrs(self, key: str) -> bool:
        """
        Determine whether the information will be included in json or not. 
        """
        avoid = { "mod" }
        return key not in avoid 

    
    def _serialize_tensor(self, key: str, val: torch.Tensor, node_name: str) -> JSON_TYPE:
        arr = self._to_numpy(val)
        weight_name = f"{node_name}_{key}_{self._weight_counter}"
        self._weight_counter += 1
        return self._export_tensor(arr, weight_name)


    def _serialize_attr(self, key: str, val: Any, node_name: str) -> JSON_TYPE:
        """Convert a single attribute value to JSON-serializable form."""
        # Handle torch.Tensor -> export as binary
        # if not self._check_is_need_attrs(key):
        #     return 

        if isinstance(val, torch.Tensor):
            return self._serialize_tensor(key, val, node_name) 

        # Handle tuple -> convert to list
        if isinstance(val, tuple):
            return list(val)

        # Handle basic JSON types
        if isinstance(val, (str, bool, int, float, type(None))):
            return val

        # Handle list
        if isinstance(val, list):
            return [self._serialize_attr(key, item, node_name) for item in val]
            # return [ val for val in lowered if val is not None ]

        # Fallback: convert to string
        # raise Exception(f'[ERROR]: {type(val)} is not supported. key -> {key} / val -> {val}')
        return str(val)

    def _is_model_parameter(self, v: Value) -> bool:
        """Check if a Value represents a model parameter (weight/bias/buffer).

        Model parameters:
        - Have data (is_const() is True)
        - Have no producer (not created by a node)
        - Are not graph inputs
        - Names typically contain 'weight', 'bias', 'running_mean', etc.
        """
        if not v.is_const():
            return False
        if v.producer is not None:
            return False
        # Model params have hierarchical names like 'layer1.0.conv1.weight'
        name_parts = v.name.split('.')
        if len(name_parts) > 0:
            last_part = name_parts[-1]
            if last_part in WEIGHT_ATTRS or 'num_batches_tracked' in last_part:
                return True
        return False

    # ---------- translation functions ----------

    def _value_to_json(self, v: Value, graph_inputs: set) -> Dict[str, JSON_TYPE] | None:
        """Convert Value to JSON. Returns None if Value should be skipped."""
        # Skip model parameters - they are exported via Node.attrs
        if self._is_model_parameter(v):
            return None

        # Export graph constants (created during rewrites, not model params)
        if v.is_const():
            return self._export_constant(v)

        # Regular Value (activation) - just metadata
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
            assert val is not None, f'[ERROR]: value of {key} is None'
            serialized_attrs[key] = self._serialize_attr(key, val, n.raw_name)

        return {
            "op_type": n.op_type,
            "name": n.raw_name,
            "inputs": [v.name for v in n.inputs],
            "outputs": [v.name for v in n.outputs],
            "attrs": serialized_attrs,
        }

    # ---------- public ----------

    def show_nodes(self) -> None:
        print(f'\n\nNode information after lowering')
        for node in self.nodes:
            print(node)
        return 


    def export(self, graph: Graph, fname: str = "graph.json") -> None:
        graph_input_names = {v.name for v in graph.inputs}

        # Build values dict, skipping model parameters
        values_json = {}
        for name, v in graph.values.items():
            v_json = self._value_to_json(v, graph_input_names)
            if v_json is not None:
                values_json[name] = v_json

        graph_json = {
            "inputs": [v.name for v in graph.inputs],
            "outputs": [v.name for v in graph.outputs],
            "values": values_json,
            "nodes": [self._node_to_json(n) for n in graph.nodes],
        }

        out_path = os.path.join(self.out_dir, fname)
        with open(out_path, "w") as f:
            json.dump(graph_json, f, indent=2)

        self.nodes = graph_json["nodes"]
        print(f"[LOG]: exported graph → {out_path}")
        return
