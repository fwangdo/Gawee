from __future__ import annotations
from typing import *
import numpy as np

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import ADD, CONV, GEMM, MUL, REDUCE_MEAN, RELU, RESHAPE
from gawee_ir.passes.errors import *
from gawee_ir.constant.passes import *


def _is_const(v: Value) -> bool:
    return v.data is not None # TODO: check whether it is valid or not. 


def _as_array(v: Value) -> np.ndarray:
    # convert Value into numpy array. 
    assert v.data is not None
    return v.data


def _all_zero(v: Value) -> bool:
    if v.data is None:
        return False
    return bool(np.all(v.data == 0))


def _all_one(v: Value) -> bool:
    if v.data is None:
        return False
    return bool(np.all(v.data == 1))


class ConstantFolding:
    """Constant folding / local algebraic simplifications (lightweight).

    - Folds only cheap ops when all inputs are const:
        Add / Mul / Relu / Reshape / ReduceMean
      (NO Conv/MaxPool/Gemm evaluation here)

    - Performs local identities for Add/Mul even if not all-const:
        Add(x,0) -> x
        Mul(x,1) -> x

    - Performs local parameter folding:
        Conv -> Add(const)  => fold into Conv bias
        Gemm -> Add(const)  => fold into Gemm bias
    """

    @classmethod
    def run(cls, g: Graph) -> bool:
        changed = False

        for n in list(g.nodes):
            if not n.outputs:
                continue

            op = n.op_type

            # 1) fold if all inputs const (light ops only)
            if op in {ADD, MUL, RELU, RESHAPE, REDUCE_MEAN}:
                # note that, if the op is fused, it should be absorbed by predecessor. 
                changed |= cls._fold_if_all_const(g, n)
                # print(f'folded -> {cls._fold_if_all_const(g, n)}')
                # node might be removed
                if n not in g.nodes:
                    continue

            # 2) local simplifications / param folding
            if op == ADD:
                changed |= cls._simplify_add(g, n)
                # print(f'simp add res -> {cls._simplify_add(g, n)}')
            elif op == MUL:
                changed |= cls._simplify_mul(g, n)
                # print(f'simp add res -> {cls._simplify_mul(g, n)}')

        return changed

    # ----------------------------
    # constant folding (cheap ops)
    # ----------------------------

    @classmethod
    def _fold_if_all_const(cls, g: Graph, n: Node) -> bool:
        if len(n.outputs) != 1:
            return False

        op = n.op_type
        out: np.ndarray | None = None

        if op == ADD:
            if len(n.inputs) != 2 or not (_is_const(n.inputs[0]) and _is_const(n.inputs[1])):
                return False
            out = _as_array(n.inputs[0]) + _as_array(n.inputs[1])
        elif op == MUL:
            if len(n.inputs) != 2 or not (_is_const(n.inputs[0]) and _is_const(n.inputs[1])):
                return False
            out = _as_array(n.inputs[0]) * _as_array(n.inputs[1])
        elif op == RELU:
            if len(n.inputs) != 1 or not _is_const(n.inputs[0]):
                return False
            out = np.maximum(_as_array(n.inputs[0]), 0)
        elif op == REDUCE_MEAN: # operation to reduction dimensions by average.  
            # TODO 
            raise NotImplementedError(CONSTANT_FOLDING, REDUCE_MEAN) 
        elif op == RESHAPE:
            raise NotImplementedError(CONSTANT_FOLDING, RESHAPE)

        if out is None:
            return False

        const_v = g.make_const(np.asarray(out))
        g.replace_all_uses(n.outputs[0], const_v)
        g.remove_node(n)
        return True

    # ----------------------------
    # simplifications (non-const)
    # ----------------------------

    @classmethod
    def _simplify_add(cls, g: Graph, n: Node) -> bool:
        if len(n.inputs) != 2 or len(n.outputs) != 1:
            return False

        a, b = n.inputs

        # canonicalize (x, const)
        if _is_const(a) and not _is_const(b):
            a, b = b, a

        # Add(x, 0) => x
        if _is_const(b) and _all_zero(b):
            g.replace_all_uses(n.outputs[0], a)
            g.remove_node(n)
            return True

        # Conv/Gemm bias folding: (producer(x) + const) # TODO. 
        return False

    @classmethod
    def _simplify_mul(cls, g: Graph, n: Node) -> bool:
        if len(n.inputs) != 2 or len(n.outputs) != 1:
            return False

        a, b = n.inputs

        # canonicalize (x, const)
        if _is_const(a) and not _is_const(b):
            a, b = b, a

        # Mul(x, 1) => x
        if _is_const(b) and _all_one(b):
            g.replace_all_uses(n.outputs[0], a)
            g.remove_node(n)
            return True

        return False