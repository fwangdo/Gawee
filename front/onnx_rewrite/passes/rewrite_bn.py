from __future__ import annotations
from typing     import List, Tuple 
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper
import math

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


    def _make_reshape_mul_add(self, x_name: str, final_out: str,
                               scale_factor: np.ndarray, bias_factor: np.ndarray) -> None:
        """Create Reshape(sf) → Mul → Reshape(bf) → Add node sequence.

        Reshapes [C] factors to [1,C,1,1] for 4D broadcasting,
        then applies x * scale_factor + bias_factor.

        Args:
            x_name: Input tensor name (BN's original input).
            final_out: Output tensor name (BN's original output).
            scale_factor: Precomputed scale factor [C].
            bias_factor: Precomputed bias factor [C].
        """
        graph = self.graph 
        sf_name = f"{prefix}_scale_factor"
        bf_name = f"{prefix}_bias_factor"
        self.add_init(graph, sf_name, scale_factor)
        self.add_init(graph, bf_name, bias_factor)

        # Reshape for 4D broadcasting: [C] -> [1, C, 1, 1]
        reshape_shape = np.array([1, -1, 1, 1], dtype=np.int64)
        sf_shape_name = f"{prefix}_sf_shape"
        bf_shape_name = f"{prefix}_bf_shape"
        self.add_init(graph, sf_shape_name, reshape_shape)
        self.add_init(graph, bf_shape_name, reshape_shape.copy())

        sf_reshaped = f"{prefix}_sf_reshaped"
        bf_reshaped = f"{prefix}_bf_reshaped"
        mul_out = f"{prefix}_mul_out"

        cls._new_nodes = [
            helper.make_node(cons.OP_RESHAPE, [sf_name, sf_shape_name], [sf_reshaped],
                             name=f"{prefix}_reshape_sf"),
            helper.make_node(cons.OP_RESHAPE, [bf_name, bf_shape_name], [bf_reshaped],
                             name=f"{prefix}_reshape_bf"),
            helper.make_node(cons.OP_MUL, [x_name, sf_reshaped], [mul_out],
                             name=f"{prefix}_mul"),
            helper.make_node(cons.OP_ADD, [mul_out, bf_reshaped], [final_out],
                             name=f"{prefix}_add"),
        ]

        return


    def _rewrite_bn_alone(self, node: onnx.NodeProto) -> None:
        x_name = node.input[0]
        scale = self.init_map.get(node.input[1])
        bias = self.init_map.get(node.input[2])
        mean = self.init_map.get(node.input[3])
        var = self.init_map.get(node.input[4])

        if any(v is None for v in [scale, bias, mean, var]):
            return 
        assert scale is not None and bias is not None and mean is not None and var is not None

        eps = self.eps
        for attr in node.attribute:
            if attr.name == cons.EPS:
                eps = float(attr.f)

        scale_factor, bias_factor = self._compute_factors(scale, bias, mean, var, eps)
        self._make_reshape_mul_add(x_name, node.output[0], scale_factor, bias_factor)

        self.nodes_to_remove.append(node)
        self.log.append(
            f" - BatchNormalization({node.name}) is converted to "
        )
        
        return 

    def _rewrite_conv_bn(self, node: onnx.NodeProto) -> None: 
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

            pred = self.producer_by_output.get(node.input[0], None)
            if pred is not None and pred in (cons.OP_CONV, cons.OP_CONV_TRANSPOSE): 
                self._rewrite_conv_bn(node)
            else: 
                self._rewrite_bn_alone(node)

        self.remove_marked_nodes()
        return model, self.log
