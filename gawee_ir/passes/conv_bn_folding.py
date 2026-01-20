# gawee_ir/passes/conv_bn_folding.py

from __future__ import annotations
from typing import *
import numpy as np
from gawee_ir.graph import Graph, Node, Value
from gawee_ir.constant.ops import *
from gawee_ir.constant.value import *


class ConvBNFolding:
    """
    Fold: Conv -> BatchNormalization  (in inference mode)
    into: Conv' with updated (W', B')
    """
    @classmethod 
    def _check_is_conv_bn(cls, x: Value) -> bool:
        conv = x.producer
        if conv is None or conv.op_type != CONV:
            return False
        return True 


    @classmethod
    def run(cls, g: Graph) -> bool:
        changed = False

        # folding for all nodes. 
        for bn in list(g.nodes):
            # check batch normalization condition. 
            print(f'[LOG]: op_type -> {bn.op_type}')
            if bn.op_type != BATCH_NORM:
                continue
            if len(bn.inputs) < 5:
                continue

            # considering only conv + bn. 
            x = bn.inputs[0]
            conv = x.producer
            if conv is None or conv.op_type != CONV:
                continue

            # log, oeprator fusion is ready to apply. 
            # print(f'[LOG]: conv + bn -> {bn}')

            # BN params
            scale, bias, mean, var = bn.inputs[1:5]
            eps = float(bn.attrs.get(EPS, 1e-5))

            if not (scale.is_const() and bias.is_const() and mean.is_const() and var.is_const()):
                # print(f'[ERROR]: There are some concepts which are not const ')
                continue

            # Conv params
            if len(conv.inputs) < 2:
                # print(f'[ERROR]: conv input -> {conv.inputs}')
                continue
            W = conv.inputs[1] # conv weight  
            B = conv.inputs[2] if len(conv.inputs) >= 3 else None # conv bias. 

            if not W.is_const():
                # print(f'[ERROR]: W -> {W}')
                continue
            if B is not None and (not B.is_const()):
                # print(f'[ERROR]: B -> {B}')
                continue

            W_arr = W.data
            if W_arr is None or W_arr.ndim != 4:
                continue

            Cout = W_arr.shape[0]
            if scale.data is None or scale.data.shape[0] != Cout:
                continue

            # ----- compute folding -----
            gamma = scale.data.astype(np.float32)
            beta  = bias.data.astype(np.float32) # type: ignore 
            mu    = mean.data.astype(np.float32) # type: ignore
            var_  = var.data.astype(np.float32)  # type: ignore

            inv_std = 1.0 / np.sqrt(var_ + eps)         
            # a means ( W / std )
            a = gamma * inv_std                         

            Wf = W_arr.astype(np.float32) * a.reshape(Cout, 1, 1, 1)

            if B is None:
                b0 = np.zeros((Cout,), dtype=np.float32)
            else:
                b0 = B.data.astype(np.float32).reshape(Cout) # type: ignore

            Bf = (b0 - mu) * a + beta                     # [Cout]

            newW = Value(
                name=W.name + "_folded",
                shape=list(Wf.shape),
                dtype="float32",
                data=Wf,
            )
            newB = Value(
                name=(B.name + "_folded") if B else (conv.name or "conv") + "_bias_folded",
                shape=list(Bf.shape),
                dtype="float32",
                data=Bf,
            )

            # graph.values 등록 (이름으로 관리하므로 충돌만 피하면 됨)
            g.values[newW.name] = newW
            g.values[newB.name] = newB

            # ----- rewrite conv inputs (bias 유무 분기 필수) -----
            # replacement. 
            conv.inputs[1] = newW
            if len(conv.inputs) == 2:
                conv.inputs.append(newB)
            elif len(conv.inputs) == 3:
                conv.inputs[2] = newB
            else:
                # Conv input arity invariant violation
                raise RuntimeError(f"Invalid Conv inputs: {len(conv.inputs)} for {conv}")

            # ----- reroute outputs: BN output becomes Conv output -----
            # pattern: conv_out -> bn -> bn_out
            # 우리가 원하는 것: conv -> bn_out
            if not conv.outputs or conv.outputs[0] is not x:
                # 그래프가 예상 패턴이 아니면 스킵 (보수적으로)
                continue

            # replacement part. 
            # detach bn from x consumer list
            if bn in x.consumers:
                x.consumers.remove(bn)

            # conv outputs replaced by bn outputs
            conv.outputs = bn.outputs
            for outv in bn.outputs:
                outv.producer = conv

            # remove BN node
            g.remove_node(bn)
            changed = True

        return changed
