from __future__ import annotations
from typing     import * 

from gawee_ir.graph import * 
from gawee_ir.constant.ops import *


class ShapeInference:

    @classmethod
    def run(cls, g: Graph) -> None:
        # nodes are assumed topologically ordered
        for n in g.nodes:
            cls._infer_node(n)

    @classmethod
    def _infer_node(cls, n: Node) -> None:
        op = n.op_type

        if op in { RELU, SIGMOID, TANH, ID, BATCH_NORM }:
            # elementwise / shape-preserving
            cls._propagate_same(n)
        elif op in { ADD, MUL, SUB, DIV }:
            # broadcast elementwise (simplified: same shape)
            cls._propagate_same(n)
        elif op == CONV:
            cls._infer_conv(n)
        elif op in { MATMUL, GEMM }:
            cls._infer_matmul(n)
        elif op == RESHAPE:
            cls._infer_reshape(n)
        elif op == TRANS:
            cls._infer_transpose(n)
        elif op in { MAXPOOL, AVGPOOL }:
            pass
        elif op == FLATTEN:
            pass
        elif op in { GETATTR, GETITEM }:
            pass
        elif op == CAT:
            pass
        elif op == INTERPOLATE:
            pass
        else:
            # unknown op: do not crash, leave shapes as-is
            print(f'op type -> {op}')
            pass

        return 

    # ---------------- helpers ----------------

    @staticmethod
    def _propagate_same(n: Node) -> None:
        if len(n.inputs) == 0:
            return

        # normally, input should be pone. 
        in_shape = n.inputs[0].shape
        if in_shape is None:
            return

        # normally, output should be one. 
        for out in n.outputs:
            out.shape = list(in_shape)
        return 


    @staticmethod
    def _infer_conv(n: Node) -> None:
        # inputs: X[N,Cin,H,W], W[Cout,Cin/groups,Kh,Kw], (Bias)
        if len(n.inputs) < 2: # it should have input and weight at least.
            return

        x, w = n.inputs[0], n.inputs[1]
        if x.shape is None or w.shape is None:
            return

        N, Cin, H, W = x.shape
        Cout, _, Kh, Kw = w.shape

        pads = n.attrs.get("pads", [0, 0, 0, 0])
        strides = n.attrs.get("strides", [1, 1])

        pad_h = pads[0] + pads[2]
        pad_w = pads[1] + pads[3]
        sh, sw = strides

        # standard method to find out output size. 
        Hout = (H + pad_h - Kh) // sh + 1
        Wout = (W + pad_w - Kw) // sw + 1

        for out in n.outputs:
            # n is the batch size. 
            out.shape = [N, Cout, Hout, Wout]

        return 


    @staticmethod
    def _infer_matmul(n: Node) -> None:
        if len(n.inputs) < 2:
            return
        a, b = n.inputs[0], n.inputs[1]
        if a.shape is None or b.shape is None:
            return

        # [M,K] x [K,N] -> [M,N] (batch ignored)
        M, K1 = a.shape[-2], a.shape[-1]
        K2, N = b.shape[-2], b.shape[-1]
        if K1 != K2:
            return

        out_shape = list(a.shape[:-2]) + [M, N]
        for out in n.outputs:
            out.shape = out_shape

        return 


    @staticmethod
    def _infer_reshape(n: Node) -> None:
        if not n.inputs:
            return
        # shape usually comes from initializer; assume already resolved
        pass
        return 


    @staticmethod
    def _infer_transpose(n: Node) -> None:
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        perm = n.attrs.get("perm")
        if perm is None:
            return

        out_shape = [x.shape[i] for i in perm]
        for out in n.outputs:
            out.shape = out_shape
        return 
