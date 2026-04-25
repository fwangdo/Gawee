from __future__ import annotations
from typing     import List, Tuple 
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class BnNodeContext:
    """Cached metadata for one BatchNormalization node.

    `pred` is the node that produces `input_name`, if any. Conv-BN fusion should
    only use it when `pred.op_type` is Conv/ConvTranspose and the produced tensor
    has this BN as its only consumer.
    """

    prefix: str
    node: onnx.NodeProto
    input_name: str
    output_name: str
    scale: np.ndarray
    bias: np.ndarray
    mean: np.ndarray
    var: np.ndarray
    eps: float
    pred: onnx.NodeProto | None
    pred_consumers: list[onnx.NodeProto]


class RewriteBN(Folder):
    """Convert BatchNormalization → Mul + Add.

    BN formula: y = (x - mean) / sqrt(var + eps) * scale + bias
    Rearranged: y = x * scale_factor + bias_factor
      where scale_factor = scale / sqrt(var + eps)
            bias_factor  = bias - mean * scale_factor

    Both factors are [C]-shaped, reshaped to [1,C,1,1] for 4D broadcasting.
    """

    def _build_context(self, node: onnx.NodeProto) -> BnNodeContext:
        """Collect cached metadata for one BatchNormalization node.

        Args:
            node: BatchNormalization node currently being inspected.

        Returns:
            Context object containing BN parameters, tensor names, epsilon,
            producer node of BN input, and consumers of the producer output.
        """
        input_name = node.input[0]
        output_name = node.output[0]

        scale = self.init_map.get(node.input[1])
        bias = self.init_map.get(node.input[2])
        mean = self.init_map.get(node.input[3])
        var = self.init_map.get(node.input[4])
        if any(value is None for value in (scale, bias, mean, var)):
            raise ValueError(f"BatchNormalization({self.get_prefix(node)}) has non-initializer parameters")

        eps = self.eps
        for attr in node.attribute:
            if attr.name == "epsilon":
                eps = float(attr.f)

        pred = self.get_producer(input_name)
        pred_consumers = self.get_consumers(pred.output[0]) if pred is not None and pred.output else []

        assert scale is not None and bias is not None and mean is not None and var is not None
        return BnNodeContext(
            prefix=self.get_prefix(node),
            node=node,
            input_name=input_name,
            output_name=output_name,
            scale=scale,
            bias=bias,
            mean=mean,
            var=var,
            eps=eps,
            pred=pred,
            pred_consumers=pred_consumers,
        )


    def _compute_factors(
        self, scale: np.ndarray, bias: np.ndarray,
        mean: np.ndarray, var: np.ndarray, eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute scale_factor and bias_factor from BN parameters.

        Args:
            scale: BN scale parameter [C].
            bias: BN bias parameter [C].
            mean: BN running mean [C].
            var: BN running variance [C].
            eps: Epsilon for numerical stability.

        Returns:
            (scale_factor, bias_factor) as float32 arrays.
        """
        scale_factor = scale / np.sqrt(var + eps)
        bias_factor = bias - mean * scale_factor

        return scale_factor.astype(np.float32), bias_factor.astype(np.float32)


    def _make_conv(
        self,
        context: BnNodeContext,
        scale_factor: np.ndarray,
        bias_factor: np.ndarray,
    ) -> List[onnx.NodeProto]:
        """Build a grouped 1x1 Conv equivalent to BatchNormalization.

        BN after factorization is channel-wise affine:
          y[:, c, h, w] = x[:, c, h, w] * scale_factor[c] + bias_factor[c]

        A depthwise 1x1 Conv expresses that directly:
          weight shape = [C, 1, 1, 1]
          bias shape   = [C]
          group        = C
        """
        new_nodes: List[onnx.NodeProto] = []
        graph = self.graph

        channel_count = int(scale_factor.reshape(-1).shape[0])
        conv_weight = scale_factor.reshape(channel_count, 1, 1, 1).astype(np.float32)
        conv_bias = bias_factor.reshape(channel_count).astype(np.float32)

        conv_weight_name = self.tensor_name(context.prefix, "conv_weight")
        conv_bias_name = self.tensor_name(context.prefix, "conv_bias")
        self.add_init(graph, conv_weight_name, conv_weight)
        self.add_init(graph, conv_bias_name, conv_bias)

        new_nodes.append(
            helper.make_node(
                cons.OP_CONV,
                [context.input_name, conv_weight_name, conv_bias_name],
                [context.output_name],
                name=self.node_name(context.prefix, "conv"),
                kernel_shape=[1, 1],
                group=channel_count,
            )
        )

        return new_nodes 


    def _rewrite_bn_alone(self, node: onnx.NodeProto, context: BnNodeContext, graph: onnx.GraphProto) -> None:
        scale_factor, bias_factor = self._compute_factors(context.scale, context.bias, context.mean, context.var, context.eps)
        new_nodes = self._make_conv(context, scale_factor, bias_factor)

        self.replace_node(node, new_nodes)
        self.log.append(
            f" - BatchNormalization({context.prefix}) is rewritten as Conv({new_nodes[0].name})"
        )
        
        return 

    @staticmethod
    def _get_conv_group(node: onnx.NodeProto) -> int:
        for attr in node.attribute:
            if attr.name == "group":
                return int(attr.i)
        return 1

    def _prepare_weight_fusion(
        self,
        context: BnNodeContext,
        expected_op_type: str,
    ) -> tuple[onnx.NodeProto, np.ndarray, int, np.ndarray, np.ndarray] | None:
        pred = context.pred
        if pred is None or pred.op_type != expected_op_type:
            return None

        op_label = expected_op_type
        if len(context.pred_consumers) != 1:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"({op_label} output has multiple consumers)"
            )
            return None

        if len(pred.input) < 2:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"({op_label} has no weight input)"
            )
            return None

        weight_name = pred.input[1]
        weight = self.init_map.get(weight_name)
        if weight is None:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"({op_label} weight is dynamic)"
            )
            return None

        group = self._get_conv_group(pred)
        scale_factor, bias_factor = self._compute_factors(
            context.scale,
            context.bias,
            context.mean,
            context.var,
            context.eps,
        )
        scale = scale_factor.reshape(-1).astype(np.float32)
        bias = bias_factor.reshape(-1).astype(np.float32)
        return pred, weight, group, scale, bias

    def _build_fused_bias(
        self,
        context: BnNodeContext,
        pred: onnx.NodeProto,
        scale: np.ndarray,
        bias: np.ndarray,
    ) -> np.ndarray | None:
        if len(pred.input) > 2 and pred.input[2]:
            conv_bias = self.init_map.get(pred.input[2])
            if conv_bias is None:
                self.log.append(
                    f" - BatchNormalization({context.prefix}) kept as BN "
                    f"({pred.op_type} bias is dynamic)"
                )
                return None
            return conv_bias.reshape(-1).astype(np.float32) * scale + bias
        return bias

    def _make_fused_node(
        self,
        context: BnNodeContext,
        graph: onnx.GraphProto,
        pred: onnx.NodeProto,
        fused_weight: np.ndarray,
        fused_bias: np.ndarray,
        weight_role: str,
        bias_role: str,
        node_role: str,
    ) -> list[onnx.NodeProto]:
        fused_weight_name = self.tensor_name(context.prefix, weight_role)
        fused_bias_name = self.tensor_name(context.prefix, bias_role)
        self.add_init(graph, fused_weight_name, fused_weight.astype(np.float32))
        self.add_init(graph, fused_bias_name, fused_bias.astype(np.float32))

        fused_inputs = [pred.input[0], fused_weight_name, fused_bias_name]
        fused_node = helper.make_node(
            pred.op_type,
            fused_inputs,
            [context.output_name],
            name=self.node_name(context.prefix, node_role),
        )
        fused_node.attribute.extend(pred.attribute)
        return [fused_node]

    def _rewrite_fused_predecessor(
        self,
        node: onnx.NodeProto,
        context: BnNodeContext,
        new_nodes: list[onnx.NodeProto] | None,
    ) -> None:
        if new_nodes is None:
            return

        assert context.pred is not None
        self.replace_node(context.pred, new_nodes)
        self.mark_for_removal(node)
        self.log.append(
            f" - {context.pred.op_type}({context.pred.name}) + BatchNormalization({context.prefix}) "
            f"is fused into {new_nodes[0].op_type}({new_nodes[0].name})"
        )


    def _make_conv_bn(
        self,
        context: BnNodeContext,
        graph: onnx.GraphProto,
        scale_factor: np.ndarray,
        bias_factor: np.ndarray,
    ) -> List[onnx.NodeProto] | None:
        """Fuse Conv followed by BatchNormalization.

        ONNX Conv weight shape is:
          [C_out, C_in / group, kH, kW]

        BatchNormalization scales Conv output channels, so the scale belongs to
        the first weight axis, split per group when group > 1.
        """
        fusion = self._prepare_weight_fusion(context, cons.OP_CONV)
        if fusion is None:
            return None
        pred, weight, group, scale, bias = fusion
        output_channels = int(weight.shape[0])
        input_channels_per_group = int(weight.shape[1])

        if output_channels != scale.shape[0]:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"(Conv output channels {output_channels} != BN channels {scale.shape[0]})"
            )
            return None
        if output_channels % group != 0:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"(Conv output channels {output_channels} not divisible by group {group})"
            )
            return None

        output_channels_per_group = output_channels // group
        grouped_weight = weight.reshape(
            group,
            output_channels_per_group,
            input_channels_per_group,
            *weight.shape[2:],
        )
        grouped_scale = scale.reshape(group, output_channels_per_group)
        fused_weight = (
            grouped_weight * grouped_scale[:, :, *([None] * (weight.ndim - 1))]
        ).reshape(weight.shape).astype(np.float32)

        fused_bias = self._build_fused_bias(context, pred, scale, bias)
        if fused_bias is None:
            return None

        return self._make_fused_node(
            context=context,
            graph=graph,
            pred=pred,
            fused_weight=fused_weight,
            fused_bias=fused_bias,
            weight_role="conv_weight",
            bias_role="conv_bias",
            node_role="conv",
        )


    def _make_conv_transpose_bn(
        self,
        context: BnNodeContext,
        graph: onnx.GraphProto,
        scale_factor: np.ndarray,
        bias_factor: np.ndarray,
    ) -> List[onnx.NodeProto] | None:
        """Fuse ConvTranspose followed by BatchNormalization.

        ONNX ConvTranspose weight shape is:
          [C_in, C_out / group, kH, kW]

        BatchNormalization scales ConvTranspose output channels, so the scale
        belongs to the second weight axis, split per group when group > 1.
        """
        fusion = self._prepare_weight_fusion(context, cons.OP_CONV_TRANSPOSE)
        if fusion is None:
            return None
        pred, weight, group, scale, bias = fusion
        input_channels, output_channels_per_group = weight.shape[:2]
        output_channels = output_channels_per_group * group

        if output_channels != scale.shape[0]:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"(ConvTranspose output channels {output_channels} != BN channels {scale.shape[0]})"
            )
            return None
        if input_channels % group != 0:
            self.log.append(
                f" - BatchNormalization({context.prefix}) kept as BN "
                f"(ConvTranspose input channels {input_channels} not divisible by group {group})"
            )
            return None

        input_channels_per_group = input_channels // group
        grouped_weight = weight.reshape(
            group,
            input_channels_per_group,
            output_channels_per_group,
            *weight.shape[2:],
        )
        grouped_scale = scale.reshape(group, output_channels_per_group)
        fused_weight = (
            grouped_weight * grouped_scale[:, None, :, *([None] * (weight.ndim - 2))]
        ).reshape(weight.shape).astype(np.float32)

        fused_bias = self._build_fused_bias(context, pred, scale, bias)
        if fused_bias is None:
            return None

        return self._make_fused_node(
            context=context,
            graph=graph,
            pred=pred,
            fused_weight=fused_weight,
            fused_bias=fused_bias,
            weight_role="conv_transpose_weight",
            bias_role="conv_transpose_bias",
            node_role="conv_transpose",
        )


    def _rewrite_conv_transpose_bn(
        self,
        node: onnx.NodeProto,
        context: BnNodeContext,
        graph: onnx.GraphProto,
    ) -> None:
        scale_factor, bias_factor = self._compute_factors(
            context.scale,
            context.bias,
            context.mean,
            context.var,
            context.eps,
        )
        self._rewrite_fused_predecessor(
            node=node,
            context=context,
            new_nodes=self._make_conv_transpose_bn(context, graph, scale_factor, bias_factor),
        )
        return


    def _rewrite_conv_bn(
            self, 
            node: onnx.NodeProto, 
            context: BnNodeContext, 
            graph: onnx.GraphProto
            ) -> None: 
        scale_factor, bias_factor = self._compute_factors(
            context.scale,
            context.bias,
            context.mean,
            context.var,
            context.eps,
        )
        self._rewrite_fused_predecessor(
            node=node,
            context=context,
            new_nodes=self._make_conv_bn(context, graph, scale_factor, bias_factor),
        )
        return


    def run(self, model: onnx.ModelProto) -> Tuple[onnx.ModelProto, List[str]]:
        """Convert all BatchNormalization nodes to Mul + Add.

        Args:
            model: ONNX model to optimize.

        Returns:
            (model, log): Modified model and list of log messages.
        """
        self.prepare(model)

        for node in self.graph.node:
            if node.op_type != cons.OP_BATCH_NORMALIZATION:
                continue

            context = self._build_context(node)
            if context.pred is not None and context.pred.op_type == cons.OP_CONV_TRANSPOSE:
                self._rewrite_conv_transpose_bn(node, context, self.graph)
            elif context.pred is not None and context.pred.op_type == cons.OP_CONV: 
                self._rewrite_conv_bn(node, context, self.graph)
            else: 
                self._rewrite_bn_alone(node, context, self.graph)

        self.remove_marked_nodes()
        return model, self.log
