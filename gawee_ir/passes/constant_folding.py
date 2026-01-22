from __future__ import annotations
from typing import *
import numpy as np

from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import ADD, CONV, GEMM, MUL, REDUCE_MEAN, RELU, RESHAPE


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
                # node might be removed
                if n not in g.nodes:
                    continue

            # 2) local simplifications / param folding
            if op == ADD:
                changed |= cls._simplify_add(g, n)
            elif op == MUL:
                changed |= cls._simplify_mul(g, n)

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

        try:
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
                if len(n.inputs) != 1 or not _is_const(n.inputs[0]):
                    return False
                axes = n.attrs.get("axes", None)
                keepdims = int(n.attrs.get("keepdims", 1))
                axis = tuple(axes) if axes is not None else None
                out = np.mean(_as_array(n.inputs[0]), axis=axis, keepdims=bool(keepdims))

            elif op == RESHAPE:
                # ONNX Reshape: inputs = [data, shape]
                if len(n.inputs) < 2:
                    return False
                data_v, shape_v = n.inputs[0], n.inputs[1]
                if not (_is_const(data_v) and _is_const(shape_v)):
                    return False
                new_shape = tuple(int(x) for x in _as_array(shape_v).reshape(-1).tolist())
                out = np.reshape(_as_array(data_v), new_shape)

            else:
                return False

        except Exception:
            return False

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

        # Conv/Gemm bias folding: (producer(x) + const)
        if _is_const(b):
            prod = a.producer
            if prod is not None and prod.op_type == CONV:
                return cls._fold_add_into_conv_bias(g, prod, n, b)
            if prod is not None and prod.op_type == GEMM:
                return cls._fold_add_into_gemm_bias(g, prod, n, b)

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

    # ----------------------------
    # parameter folding: Conv/Gemm bias
    # ----------------------------

    @classmethod
    def _fold_add_into_conv_bias(cls, g: Graph, conv: Node, add: Node, cst: Value) -> bool:
        # Conv inputs: [X, W, (B?)]
        if len(conv.inputs) < 2:
            return False

        W = conv.inputs[1]
        if not _is_const(W):
            return False

        W_arr = _as_array(W)
        if W_arr.ndim != 4:
            return False

        Cout = int(W_arr.shape[0])

        try:
            bc = _as_array(cst).reshape(-1)
            if bc.size == 1:
                bc = np.full((Cout,), float(bc[0]), dtype=np.float32)
            elif bc.size != Cout:
                return False
        except Exception:
            return False

        if len(conv.inputs) >= 3:
            B = conv.inputs[2]
            if not _is_const(B):
                return False
            b0 = _as_array(B).astype(np.float32).reshape(Cout)
        else:
            b0 = np.zeros((Cout,), dtype=np.float32)

        new_b = (b0 + bc.astype(np.float32)).astype(np.float32)
        newB = g.make_const(new_b)

        if len(conv.inputs) >= 3:
            conv.inputs[2] = newB
        else:
            conv.inputs.append(newB)

        g.replace_all_uses(add.outputs[0], conv.outputs[0])
        g.remove_node(add)
        return True

    @classmethod
    def _fold_add_into_gemm_bias(cls, g: Graph, gemm: Node, add: Node, cst: Value) -> bool:
        # Gemm: inputs = [A, B, (C?)]
        if len(gemm.inputs) < 2:
            return False

        bias = gemm.inputs[2] if len(gemm.inputs) >= 3 else None
        if bias is not None and not _is_const(bias):
            return False

        try:
            bc = _as_array(cst).astype(np.float32)
            if bias is None:
                gemm.inputs.append(g.make_const(bc))
            else:
                new_bias = (_as_array(bias).astype(np.float32) + bc).astype(np.float32)
                gemm.inputs[2] = g.make_const(new_bias)
        except Exception:
            return False

        g.replace_all_uses(add.outputs[0], gemm.outputs[0])
        g.remove_node(add)
        return True


def run(graph: Graph) -> bool:
    return ConstantFolding.run(graph)
