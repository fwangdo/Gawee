from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class GemmAttributes:
    alpha: float = 1.0
    beta: float = 1.0
    trans_a: int = 0
    trans_b: int = 0


@dataclass(frozen=True)
class GemmRewritePlan:
    prefix: str
    activation_tensor_name: str
    output_tensor_name: str
    conv_weight: np.ndarray
    output_channels: int
    trans_a: int
    conv_bias: np.ndarray | None


class RewriteGemm(Folder):
    """Rewrite Gemm nodes into an equivalent 1x1 Conv chain."""

    def _parse_attributes(self, node: onnx.NodeProto) -> GemmAttributes:
        attrs = GemmAttributes()

        for attr in node.attribute:
            if attr.name == "alpha":
                attrs = GemmAttributes(
                    alpha=float(attr.f),
                    beta=attrs.beta,
                    trans_a=attrs.trans_a,
                    trans_b=attrs.trans_b,
                )
            elif attr.name == "beta":
                attrs = GemmAttributes(
                    alpha=attrs.alpha,
                    beta=float(attr.f),
                    trans_a=attrs.trans_a,
                    trans_b=attrs.trans_b,
                )
            elif attr.name == "transA":
                attrs = GemmAttributes(
                    alpha=attrs.alpha,
                    beta=attrs.beta,
                    trans_a=int(attr.i),
                    trans_b=attrs.trans_b,
                )
            elif attr.name == "transB":
                attrs = GemmAttributes(
                    alpha=attrs.alpha,
                    beta=attrs.beta,
                    trans_a=attrs.trans_a,
                    trans_b=int(attr.i),
                )

        return attrs


    @staticmethod
    def _prepare_conv_weight(weight: np.ndarray, attrs: GemmAttributes) -> np.ndarray:
        if attrs.trans_b:
            weight = weight.T
        weight = (attrs.alpha * weight).astype(np.float32)
        input_channels, output_channels = weight.shape
        return weight.T.reshape(output_channels, input_channels, 1, 1).astype(np.float32)

    def _build_plan(
        self,
        node: onnx.NodeProto,
        attrs: GemmAttributes,
    ) -> GemmRewritePlan | None:
        activation_tensor_name = node.input[0]
        weight_tensor_name = node.input[1]
        bias_tensor_name = node.input[2] if len(node.input) > 2 else None
        output_tensor_name = node.output[0]
        prefix = self.get_prefix(node)

        weight = self.init_map.get(weight_tensor_name)
        if weight is None:
            self.log.append(f" - Gemm({prefix}) has dynamic weight - kept as Gemm")
            return None

        conv_weight = self._prepare_conv_weight(weight, attrs)
        _, output_channels = (weight.T if attrs.trans_b else weight).shape

        conv_bias: np.ndarray | None = None
        if bias_tensor_name is not None:
            bias = self.init_map.get(bias_tensor_name)
            if bias is not None:
                conv_bias = (attrs.beta * bias).astype(np.float32).flatten()

        return GemmRewritePlan(
            prefix=prefix,
            activation_tensor_name=activation_tensor_name,
            output_tensor_name=output_tensor_name,
            conv_weight=conv_weight,
            output_channels=output_channels,
            trans_a=attrs.trans_a,
            conv_bias=conv_bias,
        )

    def _emit_rewrite(
        self,
        plan: GemmRewritePlan,
        graph: onnx.GraphProto,
    ) -> List[onnx.NodeProto]:
        nodes: List[onnx.NodeProto] = []

        conv_weight_name = self.tensor_name(plan.prefix, "conv_weight")
        self.add_init(graph, conv_weight_name, plan.conv_weight)

        current_activation_name = plan.activation_tensor_name
        if plan.trans_a:
            trans_a_tensor_name = self.tensor_name(plan.prefix, "trans_a")
            nodes.append(
                helper.make_node(
                    cons.OP_TRANSPOSE,
                    [plan.activation_tensor_name],
                    [trans_a_tensor_name],
                    name=self.node_name(plan.prefix, "trans_a"),
                    perm=[1, 0],
                )
            )
            current_activation_name = trans_a_tensor_name

        transpose1_tensor_name = self.tensor_name(plan.prefix, "transpose1")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [current_activation_name],
                [transpose1_tensor_name],
                name=self.node_name(plan.prefix, "transpose1"),
                perm=[1, 0],
            )
        )

        unsqueeze_axes_name = self.tensor_name(plan.prefix, "unsqueeze_axes")
        self.add_init(graph, unsqueeze_axes_name, np.array([0, 3], dtype=np.int64))
        unsqueezed_tensor_name = self.tensor_name(plan.prefix, "unsqueezed")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [transpose1_tensor_name, unsqueeze_axes_name],
                [unsqueezed_tensor_name],
                name=self.node_name(plan.prefix, "unsqueeze"),
            )
        )

        conv_input_names = [unsqueezed_tensor_name, conv_weight_name]
        if plan.conv_bias is not None:
            conv_bias_name = self.tensor_name(plan.prefix, "conv_bias")
            self.add_init(graph, conv_bias_name, plan.conv_bias)
            conv_input_names.append(conv_bias_name)

        conv_out_tensor_name = self.tensor_name(plan.prefix, "conv_out")
        nodes.append(
            helper.make_node(
                cons.OP_CONV,
                conv_input_names,
                [conv_out_tensor_name],
                name=self.node_name(plan.prefix, "conv"),
                kernel_shape=[1, 1],
            )
        )

        transpose2_tensor_name = self.tensor_name(plan.prefix, "transpose2")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [conv_out_tensor_name],
                [transpose2_tensor_name],
                name=self.node_name(plan.prefix, "transpose2"),
                perm=[0, 2, 1, 3],
            )
        )

        reshape_shape_name = self.tensor_name(plan.prefix, "reshape_shape")
        self.add_init(graph, reshape_shape_name, np.array([-1, plan.output_channels], dtype=np.int64))
        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [transpose2_tensor_name, reshape_shape_name],
                [plan.output_tensor_name],
                name=self.node_name(plan.prefix, "reshape"),
            )
        )

        return nodes

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, List[str]]:
        self.prepare(model)
        graph = self.require_graph()

        for node in list(graph.node):
            if node.op_type != cons.OP_GEMM:
                continue

            attrs = self._parse_attributes(node)
            plan = self._build_plan(node, attrs)
            if plan is None:
                continue

            new_nodes = self._emit_rewrite(plan, graph)
            self.replace_node(node, new_nodes)
            self.log.append(
                f" - Gemm({plan.prefix}) is rewritten as Conv({self.node_name(plan.prefix, 'conv')})"
            )

        self.remove_marked_nodes()

        return model, self.log
