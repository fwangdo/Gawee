from __future__ import annotations
from typing     import *

from gawee_ir.graph import *
from gawee_ir.constant.ops import *
import math


def _to_tuple(val: Any, ndim: int = 2) -> Tuple[int, ...]:
    """Convert int or tuple to tuple of specified length."""
    if isinstance(val, int):
        return tuple([val] * ndim)
    if isinstance(val, (tuple, list)):
        return tuple(val)

    # default. 
    return tuple([1] * ndim)


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
            cls._propagate_same(n)
        elif op in { ADD, MUL, SUB, DIV }:
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
        elif op == AD_AVGPOOL:
            cls._infer_ad_pool(n)
        elif op == FLATTEN:
            cls._infer_flatten(n)
        elif op == CAT:
            cls._infer_cat(n)
        elif op == INTERPOLATE:
            cls._infer_interpolate(n)
        elif op in { GETATTR, GETITEM }:
            pass
        else:
            print(f'[ShapeInference] unknown op: {op}')
            pass

        return

    # ---------------- helpers ----------------

    @staticmethod
    def _propagate_same(n: Node) -> None:
        if len(n.inputs) == 0:
            return

        in_shape = n.inputs[0].shape
        if in_shape is None:
            return

        for out in n.outputs:
            out.shape = list(in_shape)
        return


    @staticmethod
    def _infer_conv(n: Node) -> None:
        """
        Conv from PyTorch module.
        Input: only activation X[N, Cin, H, W] (for 2D)
        Attrs:
          - out_channels: int
          - kernel_size: int or tuple
          - stride: int or tuple (default: 1)
          - padding: int or tuple (default: 0)
          - dilation: int or tuple (default: 1)
          - groups: int (default: 1)
          - weight: tensor [Cout, Cin/groups, Kh, Kw]
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        in_shape = x.shape
        ndim = len(in_shape) - 2  # spatial dims (1D, 2D, or 3D)
        if ndim < 1:
            return

        # Get attrs
        out_channels = n.attrs.get("out_channels")
        if out_channels is None:
            return

        kernel_size = _to_tuple(n.attrs.get("kernel_size", 1), ndim)
        stride = _to_tuple(n.attrs.get("stride", 1), ndim)
        padding = _to_tuple(n.attrs.get("padding", 0), ndim)
        dilation = _to_tuple(n.attrs.get("dilation", 1), ndim)

        N = in_shape[0]
        # Compute output spatial dimensions
        out_spatial = []
        for i in range(ndim):
            H_in = in_shape[2 + i]
            K = kernel_size[i]
            S = stride[i]
            P = padding[i]
            D = dilation[i]
            # PyTorch formula: floor((H + 2*P - D*(K-1) - 1) / S + 1)
            H_out = (H_in + (2 * P) - (D * (K - 1) + 1)) // S + 1
            out_spatial.append(H_out)

        out_shape = [N, out_channels] + out_spatial
        for out in n.outputs:
            out.shape = out_shape
        return


    @staticmethod
    def _infer_matmul(n: Node) -> None:
        """
        Linear layer: Y = X @ W^T + b
        Attrs:
          - in_features: int
          - out_features: int
        Input X: [..., in_features]
        Output Y: [..., out_features]
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        out_features = n.attrs.get("out_features")
        if out_features is not None:
            # Linear layer
            out_shape = list(x.shape[:-1]) + [out_features]
            for out in n.outputs:
                out.shape = out_shape
            return

        # General matmul: [M,K] x [K,N] -> [M,N]
        if len(n.inputs) < 2:
            return
        a, b = n.inputs[0], n.inputs[1]
        if a.shape is None or b.shape is None:
            return

        M, K1 = a.shape[-2], a.shape[-1]
        K2, N = b.shape[-2], b.shape[-1]
        if K1 != K2:
            return # it shouldn't happen. 

        out_shape = list(a.shape[:-2]) + [M, N]
        for out in n.outputs:
            out.shape = out_shape
        return


    @staticmethod
    def _infer_reshape(n: Node) -> None:
        """
        Reshape with target shape in attrs.
        Attrs:
          - shape: target shape (list/tuple)
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        target_shape = n.attrs.get("shape")
        if target_shape is None:
            return

        target_shape = list(target_shape)
        # Handle -1 dimension
        total = 1
        for d in x.shape:
            total *= d

        neg_idx = -1
        known = 1
        for i, d in enumerate(target_shape):
            if d == -1: # -1 means surplus dimension. 
                neg_idx = i
            else:
                known *= d

        if neg_idx >= 0:
            target_shape[neg_idx] = total // known

        for out in n.outputs:
            out.shape = target_shape
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
        MaxPool / AvgPool from PyTorch.
        Attrs:
          - kernel_size: int or tuple
          - stride: int or tuple (default: kernel_size)
          - padding: int or tuple (default: 0)
          - dilation: int or tuple (default: 1, MaxPool only)
          - ceil_mode: bool (default: False)
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        in_shape = x.shape
        ndim = len(in_shape) - 2
        if ndim < 1:
            return

        kernel_size = n.attrs.get("kernel_size")
        if kernel_size is None:
            return
        kernel_size = _to_tuple(kernel_size, ndim)

        # stride defaults to kernel_size in PyTorch
        stride = n.attrs.get("stride")
        if stride is None:
            stride = kernel_size
        else:
            stride = _to_tuple(stride, ndim)

        padding = _to_tuple(n.attrs.get("padding", 0), ndim)
        dilation = _to_tuple(n.attrs.get("dilation", 1), ndim)
        ceil_mode = n.attrs.get("ceil_mode", False)

        N, C = in_shape[0], in_shape[1]
        out_spatial = []
        for i in range(ndim):
            H_in = in_shape[2 + i]
            K = kernel_size[i]
            S = stride[i]
            P = padding[i]
            D = dilation[i]

            # Effective kernel size with dilation
            eff_K = D * (K - 1) + 1

            if ceil_mode:
                H_out = math.ceil((H_in + 2 * P - eff_K) / S) + 1
                # Avoid output size larger than expected
                if (H_out - 1) * S >= H_in + P:
                    H_out -= 1
            else:
                H_out = (H_in + 2 * P - eff_K) // S + 1
            out_spatial.append(H_out)

        out_shape = [N, C] + out_spatial
        for out in n.outputs:
            out.shape = out_shape
        return


    @staticmethod
    def _infer_ad_pool(n: Node) -> None:
        """
        Adaptive Average Pooling.
        Attrs:
          - output_size: int or tuple (target spatial size)
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        in_shape = x.shape
        ndim = len(in_shape) - 2
        if ndim < 1:
            return

        output_size = n.attrs.get("output_size")
        if output_size is None:
            return

        if isinstance(output_size, int):
            output_size = [output_size] * ndim
        else:
            output_size = list(output_size)

        # Handle None (keep original dim)
        for i in range(len(output_size)):
            if output_size[i] is None:
                output_size[i] = in_shape[2 + i]

        # it propagates as-is. 
        out_shape = list(in_shape[:2]) + output_size
        for out in n.outputs:
            out.shape = out_shape
        return


    # for python operation. 
    # TODO: check. 
    @staticmethod
    def _infer_flatten(n: Node) -> None:
        """
        Flatten.
        Attrs:
          - start_dim: int (default: 1)
          - end_dim: int (default: -1)
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        shape = list(x.shape)
        ndim = len(shape)

        start_dim = n.attrs.get("start_dim", 1)
        end_dim = n.attrs.get("end_dim", -1)

        if start_dim < 0:
            start_dim = ndim + start_dim
        if end_dim < 0:
            end_dim = ndim + end_dim

        start_dim = max(0, min(start_dim, ndim - 1))
        end_dim = max(0, min(end_dim, ndim - 1))

        if start_dim > end_dim:
            return

        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= shape[i]

        out_shape = shape[:start_dim] + [flat_size] + shape[end_dim + 1:]
        for out in n.outputs:
            out.shape = out_shape
        return


    # TODO: check. 
    @staticmethod
    def _infer_cat(n: Node) -> None:
        """
        Concatenation.
        Attrs:
          - dim or axis: int (default: 0)
        """
        if not n.inputs:
            return

        shapes = []
        for inp in n.inputs:
            if inp.shape is None:
                return
            shapes.append(list(inp.shape))

        if not shapes:
            return

        axis = n.attrs.get("dim", n.attrs.get("axis", 0))
        ndim = len(shapes[0])

        if axis < 0:
            axis = ndim + axis
        if axis < 0 or axis >= ndim:
            return

        ref_shape = shapes[0]
        for s in shapes[1:]:
            if len(s) != ndim:
                return
            for i in range(ndim):
                if i != axis and s[i] != ref_shape[i]:
                    return

        concat_size = sum(s[axis] for s in shapes)
        out_shape = list(ref_shape)
        out_shape[axis] = concat_size

        for out in n.outputs:
            out.shape = out_shape
        return


    # TODO: check. 
    @staticmethod
    def _infer_interpolate(n: Node) -> None:
        """
        Interpolate / Upsample.
        Attrs:
          - size: target spatial size (tuple)
          - scale_factor: scaling factor (float or tuple)
        """
        if not n.inputs:
            return
        x = n.inputs[0]
        if x.shape is None:
            return

        in_shape = x.shape
        ndim = len(in_shape) - 2
        if ndim < 1:
            return

        size = n.attrs.get("size")
        scale_factor = n.attrs.get("scale_factor")

        if size is not None:
            if isinstance(size, int):
                out_spatial = [size] * ndim
            else:
                out_spatial = list(size)
        elif scale_factor is not None:
            if isinstance(scale_factor, (int, float)):
                scale_factor = [scale_factor] * ndim
            out_spatial = [int(in_shape[2 + i] * scale_factor[i]) for i in range(ndim)]
        else:
            return

        out_shape = list(in_shape[:2]) + out_spatial
        for out in n.outputs:
            out.shape = out_shape
        return
