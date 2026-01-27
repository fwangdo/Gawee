# Eliminate FX-level Python bookkeeping ops that are NOT real tensor compute:
#   - getattr  (typically x.shape, x.dtype, etc.)
#   - getitem  (typically shape[2], tuple unpack, etc.)
#   - interpolate 
#   - cat.  
#
# Strategy:
#   - Only eliminate when the output is NON-TENSOR in your IR (out.shape is None).
#   - Resolve to a numpy constant (0-d scalar or 1-d vector) and replace all uses.
#   - Keep tensor-indexing getitem (where output has tensor shape) untouched.

from __future__ import annotations
from typing import *

import numpy as np
import torch

from gawee_ir.graph import Graph, Node, Value, DimType
from gawee_ir.passes.folder  import Folder
from gawee_ir.constant.ops   import *
from gawee_ir.passes.errors import *


class PythonOpElimination(Folder):
    """
    Remove Python ops (getattr/getitem) by resolving them into constants.

    This pass is intentionally conservative:
      - We only eliminate nodes whose output is non-tensor (Value.shape is None).
      - Typical targets:
          * getattr(x, "shape") -> constant vector [N, C, H, W]
          * getitem(shape, 2)   -> constant scalar H
      - We do NOT eliminate tensor slicing/indexing (getitem returning a tensor),
        because those are real tensor ops.
    """

    @staticmethod
    def _is_non_tensor_value(v: Value) -> bool:
        return v.shape is None


    @staticmethod
    def _as_numpy_const(x: Any) -> np.ndarray | None:
        """
        Convert a resolved python object to np.ndarray.
        Returns None if conversion is not supported.
        """
        # torch.Size -> tuple[int, ...]
        if isinstance(x, torch.Size):
            x = tuple(int(d) for d in x)

        if isinstance(x, np.ndarray):
            return x

        if isinstance(x, (np.integer, int)):
            return np.array(int(x), dtype=np.int64)

        if isinstance(x, (np.floating, float)):
            return np.array(float(x), dtype=np.float32)

        if isinstance(x, (np.bool_, bool)):
            return np.array(bool(x), dtype=np.bool_)

        if isinstance(x, (list, tuple)):
            # best-effort: vector of ints/floats/bools
            if all(isinstance(d, (np.integer, int)) for d in x):
                return np.array([int(d) for d in x], dtype=np.int64)
            if all(isinstance(d, (np.floating, float)) for d in x):
                return np.array([float(d) for d in x], dtype=np.float32)
            if all(isinstance(d, (np.bool_, bool)) for d in x):
                return np.array([bool(d) for d in x], dtype=np.bool_)

            # mixed types -> not supported
            return None

        # strings / objects are not supported as constants in this IR
        return None

    @classmethod
    def _resolve_getattr(cls, node: Node) -> Any | DimType | None:
        """
        Try to resolve getattr into a python value.

        We primarily support:
          - x.shape  (from IR Value.shape)
          - x.dtype  (from IR Value.dtype)
        and also getattr on already-constant numpy values.
        """
        print(f'GETATTR Node -> {node}')
        if not node.inputs or not node.outputs:
            return None

        out = node.outputs[0]
        if not cls._is_non_tensor_value(out):
            # getattr returning a tensor should not be eliminated here
            return None

        raw = node.raw
        # FX: call_function(getattr, args=(obj, "shape", ...))
        if len(raw.args) < 2:
            return None
        print(f'args -> {raw.args}')        

        base_v = node.inputs[0]
        attr = raw.args[1]
        if not isinstance(attr, str):
            return # we only consider shape and dtype.  

        # 1) common: tensor Value -> shape/dtype from analysis
        if attr == "shape":
            if isinstance(base_v.shape, list):
                return list(int(d) for d in base_v.shape) # dim type 
            raise E(SHAPE, base_v.shape) 

        if attr == "dtype":
            if base_v.dtype is not None:
                return base_v.dtype
            if base_v.data is not None:
                return str(base_v.data.dtype)

        # 2) fallback: getattr on constant numpy
        if base_v.data is not None:
            try:
                return getattr(base_v.data, attr)
            except Exception:
                return 

        return 

    @classmethod
    def _resolve_getitem(cls, node: Node) -> Any | None:
        """
        Try to resolve getitem into a python value.

        Supports indexing into:
          - numpy arrays (base_v.data)
          - 1D const vectors created by this pass (also base_v.data)
        """
        print(f'GETITEM Node -> {node}')
        if not node.inputs or not node.outputs:
            return None

        out = node.outputs[0]
        if not cls._is_non_tensor_value(out):
            # getitem returning a tensor is a real op (do not eliminate)
            return None

        raw = node.raw
        # FX: call_function(operator.getitem, args=(obj, idx))
        if len(raw.args) < 2:
            return None

        base_v = node.inputs[0]
        idx = raw.args[1]

        # We only resolve when base is already a numpy constant in IR
        if base_v.data is None:
            return None

        base_obj = base_v.data

        try:
            # idx can be int / slice / tuple of slices (rare)
            return base_obj[idx]  # type: ignore[index]
        except Exception:
            return None

    @classmethod
    def run(cls, g: Graph) -> bool:
        """
        Apply elimination to a fixpoint because resolution can be chained:
          getattr(x, "shape") -> const vector
          getitem(shape_vec, 2) -> const scalar
        """
        changed_any = False

        while True:
            changed = False

            for n in list(g.nodes):
                if not n.outputs:
                    continue

                op = n.op_type

                resolved: Any | None = None
                if op == GETATTR:
                    resolved = cls._resolve_getattr(n)
                elif op == GETITEM:
                    resolved = cls._resolve_getitem(n)
                elif op == INTERPOLATE: 
                    continue # TODO 
                elif op == CAT:
                    continue # TODO 
                else:
                    continue

                if resolved is None:
                    continue

                arr = cls._as_numpy_const(resolved)
                if arr is None:
                    # Cannot materialize as numeric const in this IR
                    continue

                old_out = n.outputs[0]

                # Replace old_out with new constant
                const_v = g.make_const(arr, name=f"pyconst_{n.name}")
                g.replace_all_uses(old_out, const_v)

                # Detach node cleanly (Graph.remove_node already fixes consumers/producers)
                g.remove_node(n)

                changed = True
                changed_any = True

            if not changed:
                break

        return changed_any
