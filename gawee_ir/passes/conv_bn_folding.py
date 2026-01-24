# gawee_ir/passes/conv_bn_folding.py

from __future__ import annotations
from typing import *

import numpy as np
import torch
import torch.nn as nn

from gawee_ir.graph import Graph, Node, Value, DimType
from gawee_ir.constant.ops import CONV, BATCH_NORM
from gawee_ir.types.torch_type import *


class ConvBNFolding:
    """
    Fold:   Conv -> BatchNorm   (in inference/eval mode)
    Into:   Conv' (module weights/bias updated), BN node removed

    Current-IR assumptions:
      - Conv/BN are CALL_MODULE nodes.
      - Parameters/buffers are NOT in node.inputs; they live in n.attrs["mod"] (nn.Module).
      - Node.inputs contain only activation Values.
    """

    @staticmethod
    def _shape(v: Value) -> DimType:
        s = v.shape
        if not isinstance(s, list):
            raise Exception(f'[ERROR]: {s} should be list.')
        return s


    @staticmethod
    def _get_conv_mod(conv: Node) -> CONV_TYPE:
        m = conv.attrs.get("mod", None)
        if isinstance(m, CONV_TYPE):
            return m
        raise Exception(f'[ERROR] m is not conv, m -> {m}')


    @staticmethod
    def _get_bn_mod(bn: Node) -> BN_TYPE:
        m = bn.attrs.get("mod", None)
        if isinstance(m, BN_TYPE):
            return m
        raise Exception(f'[ERROR] m is not batch, m -> {m}')


    @classmethod
    def _is_const(cls, bn_mod) -> bool:
        res = not getattr(bn_mod, "training", False)
        return res 


    @classmethod 
    def _check_and_get_bn_mod(cls, bn: Node) -> BN_TYPE | None:
        if bn.op_type != BATCH_NORM:
            return 
        if len(bn.inputs) != 1 or len(bn.outputs) != 1:
            return 

        # BN must be eval/inference to be semantics-preserving
        bn_mod = cls._get_bn_mod(bn)
        if bn_mod is None:
            return 
        if not cls._is_const(bn_mod): 
            return 

        return bn_mod


    @classmethod
    def _check_and_get_conv_mod(cls, bn: Node) -> CONV_TYPE | None:
        x = bn.inputs[0]
        conv = x.producer
        if conv is None or conv.op_type != CONV:
            return 
        conv_mod = cls._get_conv_mod(conv)
        if conv_mod is None:
            return 
        return conv_mod


    @classmethod 
    def _check_dim_condition(cls, bn: Node) -> bool:
        # True means "valid".
        xs = cls._shape(bn.inputs[0])
        ys = cls._shape(bn.outputs[0])
        if xs is None or ys is None:
            # allow folding without shapes; still correct in eval
            return False
            # pass
        else:
            # Conv/BN in ResNet are typically NCHW (len==4) or NCL (len==3)
            if len(xs) < 3 or len(ys) < 3:
                return False
        return True 


    @classmethod 
    def _change_conv_weight(cls, conv_mod: CONV_TYPE, Wf, Bf):
        with torch.no_grad():
            conv_mod.weight.copy_(torch.from_numpy(Wf).to(device=conv_mod.weight.device, dtype=conv_mod.weight.dtype))
            if conv_mod.bias is None:
                # create bias parameter on correct device/dtype
                conv_mod.bias = torch.nn.Parameter(
                    torch.from_numpy(Bf).to(device=conv_mod.weight.device, dtype=conv_mod.weight.dtype)
                )
            else:
                conv_mod.bias.copy_(torch.from_numpy(Bf).to(device=conv_mod.bias.device, dtype=conv_mod.bias.dtype))
        return 

    @classmethod
    def run(cls, g: Graph) -> bool:
        changed = False

        for bn in list(g.nodes):
            # checking whether the operation is valid or not. 
            bn_mod = cls._check_and_get_bn_mod(bn)
            if bn_mod is None:
                continue
            conv_mod = cls._check_and_get_conv_mod(bn)
            if conv_mod is None:
                continue
            if not cls._check_dim_condition(bn):
                continue

            # ---- Extract BN params from module ----
            # gamma/beta are parameters; running_mean/var are buffers.
            # In eval mode they are constant w.r.t. input.
            assert bn_mod.weight is not None and bn_mod.bias is not None, "BN without affine not supported yet"
            gamma = bn_mod.weight.detach().cpu().numpy().astype(np.float32)         
            beta  = bn_mod.bias.detach().cpu().numpy().astype(np.float32)           
            mu    = bn_mod.running_mean.detach().cpu().numpy().astype(np.float32)    # type: ignore 
            var   = bn_mod.running_var.detach().cpu().numpy().astype(np.float32)     # type: ignore 
            eps   = float(bn_mod.eps)

            # ---- Extract Conv params from module ----
            W = conv_mod.weight.detach().cpu().numpy().astype(np.float32)
            # W shape: [Cout, Cin/groups, K...]
            Cout = int(W.shape[0])
            # print(f'Cout -> {W.shape}, gamma shape -> {gamma.shape[0]}')
            if gamma.shape[0] != Cout:
                # Typically BN follows Conv so channels should match. If not, skip.
                continue

            if conv_mod.bias is None:
                b0 = np.zeros((Cout,), dtype=np.float32)
            else:
                b0 = conv_mod.bias.detach().cpu().numpy().astype(np.float32).reshape(Cout)

            # ---- Compute folded weights/bias ----
            inv_std = 1.0 / np.sqrt(var + eps)          # [Cout]
            a = gamma * inv_std                         # [Cout]

            # Reshape a to broadcast across weight dims: [Cout, 1, 1, ...]
            a_w = a.reshape((Cout,) + (1,) * (W.ndim - 1)) # reshape for w', (Cout, 1, 1, 1) 
            Wf = W * a_w                                # w'  
            Bf = (b0 - mu) * a + beta                   # b'  

            # ---- Write back into Conv module (in-place, no_grad) ----
            cls._change_conv_weight(conv_mod, Wf, Bf)

            # ---- Graph rewrite: bypass BN ----
            # We want all uses of BN output to use Conv output instead.
            bn_out = bn.outputs[0]
            conv_out = bn.inputs[0]               # conv produces this

            g.replace_all_uses(bn_out, conv_out)

            # Detach bn from producer/consumer lists if your IR tracks them
            if bn in conv_out.consumers:
                conv_out.consumers.remove(bn)

            g.remove_node(bn)
            changed = True

        return changed
