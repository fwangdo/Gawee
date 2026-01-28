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
    def _as_numpy_const(x) -> np.ndarray:
        """
        Convert a resolved python object to np.ndarray.
        Returns None if conversion is not supported.
        """
        if isinstance(x, np.ndarray):
            return x

        if isinstance(x, (np.integer, int)):
            return np.array(int(x), dtype=np.int64)

        if isinstance(x, (np.floating, float)):
            return np.array(float(x), dtype=np.float32)

        if isinstance(x, (np.bool_, bool)):
            return np.array(bool(x), dtype=np.bool_)

        # if isinstance(x, (list, tuple)):
        if isinstance(x, list):
            # best-effort: vector of ints/floats/bools
            if all(isinstance(d, (np.integer, int)) for d in x):
                return np.array([int(d) for d in x], dtype=np.int64)
            if all(isinstance(d, (np.floating, float)) for d in x):
                return np.array([float(d) for d in x], dtype=np.float32)
            if all(isinstance(d, (np.bool_, bool)) for d in x):
                return np.array([bool(d) for d in x], dtype=np.bool_)

            # mixed types -> not supported
            raise Exception(f'[ERROR]: {x[0]} is in {type(x[0])}')

        # strings / objects are not supported as constants in this IR
        raise Exception(f'[ERROR]: x -> {x}, type -> {type(x)}')
        # return None

    @classmethod
    def _resolve_getattr(cls, node: Node) -> DimType | str:
        """
        Try to resolve getattr into a python value.

        We primarily support:
          - x.shape  (from IR Value.shape)
          - x.dtype  (from IR Value.dtype)
        and also getattr on already-constant numpy values.
        """
        # print(f'GETATTR Node -> {node}')
        if not node.inputs or not node.outputs:
            raise Exception(f'[ERROR]: inputs -> {node.inputs}, outputs -> {node.outputs}')

        out = node.outputs[0]
        if not cls._is_non_tensor_value(out): # the shape is None. 
            raise Exception(f'[ERROR] {out} shapse is None.') 

        raw = node.raw
        # FX: call_function(getattr, args=(obj, "shape", ...))
        if len(raw.args) < 2:
            raise Exception(f'[ERROR]: {raw.args} should consist of data and attribute.')

        base_v = node.inputs[0]
        attr = raw.args[1]
        if not isinstance(attr, str):
            raise Exception(f'[ERROR]: attribute should be in string. attr -> {attr} / {type(attr)}')

        # 1) common: tensor Value -> shape/dtype from analysis
        if attr == SHAPE:
            if isinstance(base_v.shape, list):
                return list(int(d) for d in base_v.shape) # dim type 
            raise PythonOpError(SHAPE, base_v.shape) 
        elif attr == DTYPE:
            if base_v.dtype is not None:
                return base_v.dtype
            raise PythonOpError(DTYPE, base_v.dtype)

        raise Exception(f'[ERROR]: {attr} is not defined yet. ')


    @classmethod
    def _resolve_getitem(cls, node: Node) -> Any | None:
        """
        Try to resolve getitem into a python value.
        This funstion supports constant value only. 

        Supports indexing into:
          - numpy arrays (base_v.data)
          - 1D const vectors created by this pass (also base_v.data)
        """
        if not node.inputs or not node.outputs:
            raise Exception(f'[ERROR]: inputs -> {node.inputs}, outputs -> {node.outputs}')

        out = node.outputs[0]
        if not cls._is_non_tensor_value(out):
            raise Exception(f'[ERROR] {out} shapse is None.') 

        raw = node.raw
        # FX: call_function(operator.getitem, args=(obj, idx))
        if len(raw.args) < 2:
            raise Exception(f'[ERROR]: {raw.args} should consist of data and attribute.')

        base_v = node.inputs[0]
        idx = raw.args[1]

        # We only resolve when base is already a numpy constant in IR
        if base_v.data is None:
            return # it cannot be expressed now.  

        base_obj = base_v.data
        # print(f'node -> {node} base -> {base_v}, data -> {base_v.data}, idx -> {idx}')

        try:
            # idx can be int / slice / tuple of slices (rare)
            return base_obj[idx]  # type: ignore[index]
        except Exception as e:
            raise Exception(f'[ERROR]: {e}') 


    @classmethod  
    def _substitute(cls, n: Node) -> DimType | str | None: 
        op = n.op_type

        if op == GETATTR:
            res = cls._resolve_getattr(n)
            # print(f'res of getattr -> {res}, type -> {type(res)}')
        elif op == GETITEM:
            res = cls._resolve_getitem(n)
        else:
            return 

        return res 


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

                resolved = cls._substitute(n)

                if resolved is None:
                    continue

                arr = cls._as_numpy_const(resolved)
                if arr is None:
                    # print(f'[LOG]: arr is None case. ')
                    # Cannot materialize as numeric const in this IR
                    continue

                old_out = n.outputs[0]

                # Replace old_out with new constant
                const_v = g.make_const(arr, name=f"pyconst_{n.name}")
                g.replace_all_uses(old_out, const_v)

                # Detach node cleanly (Graph.remove_node already fixes consumers/producers)
                # Dead code elimination automatically. 
                g.remove_node(n)
                cls.deleted_node += 1

                changed = True
                changed_any = True

            # until there is nothing to delete. 
            if not changed:
                break

        return changed_any
