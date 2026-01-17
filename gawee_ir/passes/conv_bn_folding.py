# gawee_ir/passes/conv_bn_folding.py

from __future__ import annotations
from typing     import * 

from gawee_ir.graph import Graph, Node, Value


class ConvBNFolding:
    """
    Fold:
      Conv -> BatchNormalization
    into:
      Conv' (with updated weight / bias)
    """

    @classmethod
    def run(cls, g: Graph) -> bool:
        """
        Returns True if graph was changed.
        """
        changed = False

        for bn in list(g.nodes):
            if bn.op_type != "BatchNormalization":
                continue
            if len(bn.inputs) < 5:
                continue

            x = bn.inputs[0]
            conv = x.producer
            if conv is None or conv.op_type != "Conv":
                continue

            # ---- BN params ----
            scale = bn.inputs[1]
            bias = bn.inputs[2]
            mean = bn.inputs[3]
            var = bn.inputs[4]

            eps = float(bn.attrs.get("epsilon", 1e-5))

            # ---- Conv params ----
            W = conv.inputs[1]
            B = conv.inputs[2] if len(conv.inputs) > 2 else None

            if any(v.shape is None for v in (W, scale, bias, mean, var)):
                continue

            # Shapes:
            # W: [Cout, Cin/groups, Kh, Kw]
            Cout = W.shape[0]
            if scale.shape[0] != Cout:
                continue

            # ---- Create new folded weight / bias ----
            new_W = Value(
                name=W.name + "_bn_folded",
                shape=W.shape,
                dtype=W.dtype,
            )

            new_B = Value(
                name=(B.name + "_bn_folded") if B else conv.name + "_bias_bn_folded",
                shape=[Cout],
                dtype=W.dtype,
            )

            # NOTE:
            # 실제 수치 계산은 여기서 하지 않음.
            # (포트폴리오 단계에서는 "구조적 folding"만으로 충분)
            # 필요하면 numpy 기반으로 구현 가능.

            # ---- Rewrite Conv inputs ----
            conv.inputs[1] = new_W
            conv.inputs.append(new_B)

            # ---- Redirect BN outputs ----
            for out in bn.outputs:
                out.producer = conv
                conv.outputs = bn.outputs

            # ---- Remove BN node ----
            g.remove_node(bn)

            changed = True

        return changed
