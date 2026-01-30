from __future__ import annotations
from typing     import * 

from gawee_ir.graph import * 
from gawee_ir.constant.ops import *
import math


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
            cls._infer_pool(n)
        elif op == FLATTEN:
            cls._infer_flatten(n)
        elif op == CAT:
            cls._infer_cat(n)
        elif op == INTERPOLATE:
            cls._infer_interpolate(n)
        elif op in { GETATTR, GETITEM }:
            pass #? 
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

    
    @staticmethod
    def _infer_pool(n: Node) -> None:
        """
        NCHW 2D pooling.
        attrs (typical):
          - kernel_shape: [Kh, Kw]   (required)
          - strides: [sh, sw]        (default [1,1])
          - pads: [pt, pl, pb, pr]   (default [0,0,0,0])
          - dilations: [dh, dw]      (default [1,1])  (rare but possible)
          - ceil_mode: 0/1           (default 0)
        """
        # error handling. 
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None or len(x.shape) != 4:
            return

        kernel = n.attrs.get("kernel_shape")
        if kernel is None or len(kernel) != 2:
            return
        Kh, Kw = kernel

        strides = n.attrs.get("strides", [1, 1])
        pads = n.attrs.get("pads", [0, 0, 0, 0])
        dilations = n.attrs.get("dilations", [1, 1])
        ceil_mode = n.attrs.get("ceil_mode", 0)

        if len(strides) != 2 or len(pads) != 4 or len(dilations) != 2:
            return

        N, C, H, W = x.shape
        sh, sw = strides
        pt, pl, pb, pr = pads
        dh, dw = dilations

        eff_kh = (Kh - 1) * dh + 1
        eff_kw = (Kw - 1) * dw + 1

        num_h = H + pt + pb - eff_kh
        num_w = W + pl + pr - eff_kw

        if sh <= 0 or sw <= 0:
            return

        if ceil_mode:
            Hout = math.floor((num_h + sh - 1) / sh) + 1  # ceil division
            Wout = math.floor((num_w + sw - 1) / sw) + 1
        else:
            Hout = math.floor(num_h / sh) + 1
            Wout = math.floor(num_w / sw) + 1

        for out in n.outputs:
            out.shape = [N, C, Hout, Wout]
        return

    
    @staticmethod
    def _infer_flatten(n: Node) -> None:
        return 


    @staticmethod
    def _infer_cat(n: Node) -> None:
        return 


    @staticmethod
    def _infer_interpolate(n: Node) -> None:
        return 