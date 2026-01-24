from __future__ import annotations
from typing import *

import numpy as np
import torch
import torch.nn as nn

from gawee_ir.graph            import Graph, Node, Value, DimType
from gawee_ir.constant.ops     import CONV, ADD
from gawee_ir.types.torch_type import *  # CONV_TYPE, etc.
from gawee_ir.passes.folder    import Folder


class ConvAddFolding(Folder):
    """
    Fold:   (Conv(x) + const)  -> Conv'(x)
    Where const is broadcastable to Conv output, and is per-output-channel bias.

    This is safe for inference (and training too, mathematically), but we implement
    conservatively for typical CNN patterns:
      - const shape is [C] or [1, C, 1, 1] (NCHW) or [1, C, 1] (NCL)

    This pass is applicable only when the Add operand can be represented as a per-channel constant bias.
    """

    # ---------------- helpers ----------------

    @staticmethod
    def _try_extract_channel_bias(const_arr: np.ndarray, Cout: int, out_ndim: int) -> np.ndarray | None:
        """
        Return bias vector of shape [Cout] if const is a per-channel bias broadcast.
        Supported:
          - [Cout]
          - [1, Cout, 1, 1] for NCHW
          - [1, Cout, 1]    for NCL
          - scalar -> expand to [Cout]
        """
        a = const_arr.astype(np.float32)

        # scalar
        if a.ndim == 0 or a.size == 1:
            return np.full((Cout,), float(a.reshape(-1)[0]), dtype=np.float32)

        # [Cout]
        if a.ndim == 1 and a.shape[0] == Cout:
            return a.reshape(Cout).astype(np.float32)

        # NCHW: [1, C, 1, 1]
        if out_ndim == 4 and a.ndim == 4 and a.shape == (1, Cout, 1, 1):
            return a.reshape(Cout).astype(np.float32)

        # NCL: [1, C, 1]
        if out_ndim == 3 and a.ndim == 3 and a.shape == (1, Cout, 1):
            return a.reshape(Cout).astype(np.float32)

        # Sometimes exporters produce [C,1,1] (no batch dim)
        if out_ndim == 4 and a.ndim == 3 and a.shape == (Cout, 1, 1):
            return a.reshape(Cout).astype(np.float32)

        return None

    # ---------------- main pass ----------------

    @classmethod
    def run(cls, g: Graph) -> bool:
        # TODO. 
        changed = False

        for add in list(g.nodes):
            # print(f'op types -> {add.op_type}')
            if add.op_type != ADD:
                continue
            if len(add.inputs) != 2 or len(add.outputs) != 1:
                continue

            a, b = add.inputs[0], add.inputs[1]

            # canonicalize: a = non-const activation path, b = const
            if cls._is_const_value(a) and not cls._is_const_value(b):
                a, b = b, a

            if not cls._is_const_value(b):
                continue  # we only fold Conv + const

            # producer of a must be conv
            conv = a.producer
            if conv is None or conv.op_type != CONV:
                continue

            conv_mod = cls._get_conv_mod(conv)

            # need conv output shape to know Cout
            # to guarantee add is corresponding to shape of conv. 
            if not conv.outputs:
                continue
            conv_out = conv.outputs[0]
            out_shape = cls._shape(conv_out)
            if len(out_shape) < 2:
                continue

            # Determine Cout from output tensor (preferred)
            Cout = int(out_shape[1])  # NCHW or NCL: channel dim = 1
            if Cout <= 0:
                continue

            # Extract per-channel bias vector
            const_arr = cls._as_np(b)
            bc = cls._try_extract_channel_bias(const_arr, Cout=Cout, out_ndim=len(out_shape))
            if bc is None:
                # Not a safe/recognized broadcast form -> skip
                continue

            # Get existing conv bias (or zeros)
            if conv_mod.bias is None:
                b0 = np.zeros((Cout,), dtype=np.float32)
            else:
                b0 = conv_mod.bias.detach().cpu().numpy().astype(np.float32).reshape(Cout)

            new_bias = (b0 + bc).astype(np.float32)

            # Write back into conv module (in-place)
            with torch.no_grad():
                if conv_mod.bias is None:
                    conv_mod.bias = torch.nn.Parameter(
                        torch.from_numpy(new_bias).to(device=conv_mod.weight.device, dtype=conv_mod.weight.dtype)
                    )
                else:
                    conv_mod.bias.copy_(
                        torch.from_numpy(new_bias).to(device=conv_mod.bias.device, dtype=conv_mod.bias.dtype)
                    )

            # Graph rewrite: replace uses of Add output with Conv output
            add_out = add.outputs[0]
            g.replace_all_uses(add_out, conv_out)

            # Detach consumer links if you maintain them
            if add in conv_out.consumers:
                conv_out.consumers.remove(add)

            # Remove Add node
            g.remove_node(add)
            changed = True

        return changed
