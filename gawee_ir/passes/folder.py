from __future__ import annotations
from typing import *

import numpy as np
import torch
import torch.nn as nn

from gawee_ir.graph import Graph, Node, Value, DimType
from gawee_ir.constant.ops import CONV, ADD
from gawee_ir.types.torch_type import *  # CONV_TYPE, etc.


class Folder:
    """
    Common methods for passes. 
    """

    # ---------------- helpers ----------------

    @staticmethod
    def _shape(v: Value) -> DimType:
        s = v.shape
        if not isinstance(s, list):
            raise Exception(f"[ERROR]: shape should be list, got {s}")
        return s


    @staticmethod
    def _get_conv_mod(conv: Node) -> CONV_TYPE:
        m = conv.attrs.get("mod", None)
        if isinstance(m, CONV_TYPE):
            return m
        raise Exception(f"[ERROR]: node is not conv module, mod={m}, node={conv}")


    @staticmethod
    def _get_bn_mod(bn: Node) -> BN_TYPE:
        m = bn.attrs.get("mod", None)
        if isinstance(m, BN_TYPE):
            return m
        raise Exception(f'[ERROR] m is not batch, m -> {m}')


    @staticmethod
    def _is_const_value(v: Value) -> bool:
        return v.data is not None


    @staticmethod
    def _as_np(v: Value) -> np.ndarray:
        assert v.data is not None
        return v.data