from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


ShapeValue = int | str


@dataclass(frozen=True)
class DynamicMatmulPlan:
    prefix: str
    left_name: str
    right_name: str
    output_name: str
    left_rank: int
    right_rank: int


@dataclass(frozen=True)
class StaticMatmulPlan:
    prefix: str
    activation_name: str
    activation_shape: list[ShapeValue]
    weight: np.ndarray
    output_name: str
    weight_on_right: bool


class RewriteMatmul(Folder):
    """Rewrite MatMul into supported operators."""

    def _plan_dynamic_rewrite(
        self,
        prefix: str,
        left_name: str,
        right_name: str,
        output_name: str,
    ) -> DynamicMatmulPlan | None:
        left_shape = self.shape_info.get(left_name)
        right_shape = self.shape_info.get(right_name)
        if left_shape is None or right_shape is None:
            self.log.append(f" - MatMul({prefix}) kept as MatMul (shape unknown)")
            return None

        return DynamicMatmulPlan(
            prefix=prefix,
            left_name=left_name,
            right_name=right_name,
            output_name=output_name,
            left_rank=len(left_shape),
            right_rank=len(right_shape),
        )

    def _plan_static_rewrite(
        self,
        prefix: str,
        left_name: str,
        right_name: str,
        output_name: str,
    ) -> StaticMatmulPlan | None:
        weight_on_right = right_name in self.init_map
        weight_name = right_name if weight_on_right else left_name
        activation_name = left_name if weight_on_right else right_name

        activation_shape = self.shape_info.get(activation_name)
        if activation_shape is None:
            self.log.append(f" - MatMul({prefix}) kept as MatMul (activation shape unknown)")
            return None

        squeezed_weight = self._squeeze_to_matrix(self.init_map[weight_name])
        if squeezed_weight is None:
            self.log.append(f" - MatMul({prefix}) kept as MatMul (unsupported weight shape)")
            return None

        return StaticMatmulPlan(
            prefix=prefix,
            activation_name=activation_name,
            activation_shape=activation_shape,
            weight=squeezed_weight,
            output_name=output_name,
            weight_on_right=weight_on_right,
        )

    @staticmethod
    def _squeeze_to_matrix(weight: np.ndarray) -> np.ndarray | None:
        if weight.ndim == 2:
            return weight
        if all(dim == 1 for dim in weight.shape[:-2]):
            return weight.reshape(weight.shape[-2], weight.shape[-1])
        return None

    def _emit_dynamic_rewrite(
        self,
        plan: DynamicMatmulPlan,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []

        left_axes_name = self.tensor_name(plan.prefix, "left_unsqueeze_axes")
        self.add_init(graph, left_axes_name, np.array([plan.left_rank], dtype=np.int64))
        left_unsqueezed_name = self.tensor_name(plan.prefix, "left_unsqueezed")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [plan.left_name, left_axes_name],
                [left_unsqueezed_name],
                name=self.node_name(plan.prefix, "left_unsqueeze"),
            )
        )

        right_axes_name = self.tensor_name(plan.prefix, "right_unsqueeze_axes")
        self.add_init(graph, right_axes_name, np.array([plan.right_rank - 2], dtype=np.int64))
        right_unsqueezed_name = self.tensor_name(plan.prefix, "right_unsqueezed")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [plan.right_name, right_axes_name],
                [right_unsqueezed_name],
                name=self.node_name(plan.prefix, "right_unsqueeze"),
            )
        )

        multiplied_name = self.tensor_name(plan.prefix, "broadcast_mul")
        nodes.append(
            helper.make_node(
                cons.OP_MUL,
                [left_unsqueezed_name, right_unsqueezed_name],
                [multiplied_name],
                name=self.node_name(plan.prefix, "broadcast_mul"),
            )
        )

        reduce_axis = max(plan.left_rank + 1, plan.right_rank + 1) - 2
        reduce_axes_name = self.tensor_name(plan.prefix, "reduce_sum_axes")
        self.add_init(graph, reduce_axes_name, np.array([reduce_axis], dtype=np.int64))
        nodes.append(
            helper.make_node(
                cons.OP_REDUCE_SUM,
                [multiplied_name, reduce_axes_name],
                [plan.output_name],
                name=self.node_name(plan.prefix, "reduce_sum"),
                keepdims=0,
            )
        )
        return nodes

    def _emit_static_rewrite(
        self,
        plan: StaticMatmulPlan,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto] | None:
        if plan.weight_on_right:
            return self._emit_right_static_rewrite(plan, graph)
        return self._emit_left_static_rewrite(plan, graph)

    def _emit_right_static_rewrite(
        self,
        plan: StaticMatmulPlan,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto] | None:
        input_channels, output_channels = plan.weight.shape
        conv_weight_name = self.tensor_name(plan.prefix, "conv_weight")
        conv_weight = plan.weight.T.reshape(output_channels, input_channels, 1, 1).astype(np.float32)
        self.add_init(graph, conv_weight_name, conv_weight)

        rank = len(plan.activation_shape)
        if rank == 2:
            return self._emit_matrix_conv_chain(
                prefix=plan.prefix,
                activation_name=plan.activation_name,
                conv_weight_name=conv_weight_name,
                output_name=plan.output_name,
                output_channels=output_channels,
                graph=graph,
            )
        if rank == 3:
            return self._emit_rank3_conv_chain(
                prefix=plan.prefix,
                activation_name=plan.activation_name,
                conv_weight_name=conv_weight_name,
                output_name=plan.output_name,
                output_channels=output_channels,
                graph=graph,
            )
        if rank == 4:
            return self._emit_rank4_conv_chain(
                prefix=plan.prefix,
                activation_name=plan.activation_name,
                conv_weight_name=conv_weight_name,
                output_name=plan.output_name,
                output_channels=output_channels,
                input_channels=input_channels,
                activation_shape=plan.activation_shape,
                graph=graph,
            )

        self.log.append(f" - MatMul({plan.prefix}) kept as MatMul (unsupported activation rank)")
        return None

    def _emit_left_static_rewrite(
        self,
        plan: StaticMatmulPlan,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto] | None:
        rank = len(plan.activation_shape)
        if rank not in (2, 3):
            self.log.append(f" - MatMul({plan.prefix}) kept as MatMul (left-static rank unsupported)")
            return None

        nodes: list[onnx.NodeProto] = []
        activation_permutation = [1, 0] if rank == 2 else [0, 2, 1]

        transposed_activation_name = self.tensor_name(plan.prefix, "activation_transposed")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [plan.activation_name],
                [transposed_activation_name],
                name=self.node_name(plan.prefix, "activation_transpose"),
                perm=activation_permutation,
            )
        )

        transposed_weight = plan.weight.T.astype(np.float32)
        transposed_input_channels, transposed_output_channels = transposed_weight.shape
        conv_weight_name = self.tensor_name(plan.prefix, "inner_conv_weight")
        conv_weight = (
            transposed_weight.T.reshape(transposed_output_channels, transposed_input_channels, 1, 1)
            .astype(np.float32)
        )
        self.add_init(graph, conv_weight_name, conv_weight)

        inner_output_name = self.tensor_name(plan.prefix, "inner_output")
        if rank == 2:
            inner_nodes = self._emit_matrix_conv_chain(
                prefix=f"{plan.prefix}_inner",
                activation_name=transposed_activation_name,
                conv_weight_name=conv_weight_name,
                output_name=inner_output_name,
                output_channels=transposed_output_channels,
                graph=graph,
            )
        else:
            inner_nodes = self._emit_rank3_conv_chain(
                prefix=f"{plan.prefix}_inner",
                activation_name=transposed_activation_name,
                conv_weight_name=conv_weight_name,
                output_name=inner_output_name,
                output_channels=transposed_output_channels,
                graph=graph,
            )

        if inner_nodes is None:
            self.log.append(f" - MatMul({plan.prefix}) kept as MatMul (left-static inner chain failed)")
            return None

        nodes.extend(inner_nodes)
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [inner_output_name],
                [plan.output_name],
                name=self.node_name(plan.prefix, "output_transpose"),
                perm=activation_permutation,
            )
        )
        return nodes

    def _emit_matrix_conv_chain(
        self,
        prefix: str,
        activation_name: str,
        conv_weight_name: str,
        output_name: str,
        output_channels: int,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []

        input_shape_name = self.tensor_name(prefix, "input_reshape_shape")
        output_shape_name = self.tensor_name(prefix, "output_reshape_shape")
        self.add_init(graph, input_shape_name, np.array([1, 0, -1, 1], dtype=np.int64))
        self.add_init(graph, output_shape_name, np.array([-1, output_channels], dtype=np.int64))

        input_tensor_name = self.tensor_name(prefix, "input_reshaped")
        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [activation_name, input_shape_name],
                [input_tensor_name],
                name=self.node_name(prefix, "input_reshape"),
            )
        )

        conv_output_name = self.tensor_name(prefix, "conv_output")
        nodes.append(
            helper.make_node(
                cons.OP_CONV,
                [input_tensor_name, conv_weight_name],
                [conv_output_name],
                name=self.node_name(prefix, "conv"),
                kernel_shape=[1, 1],
            )
        )

        transposed_output_name = self.tensor_name(prefix, "output_transposed")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [conv_output_name],
                [transposed_output_name],
                name=self.node_name(prefix, "output_transpose"),
                perm=[0, 2, 1, 3],
            )
        )

        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [transposed_output_name, output_shape_name],
                [output_name],
                name=self.node_name(prefix, "output_reshape"),
            )
        )
        return nodes

    def _emit_rank3_conv_chain(
        self,
        prefix: str,
        activation_name: str,
        conv_weight_name: str,
        output_name: str,
        output_channels: int,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []

        transposed_input_name = self.tensor_name(prefix, "input_transposed")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [activation_name],
                [transposed_input_name],
                name=self.node_name(prefix, "input_transpose"),
                perm=[0, 2, 1],
            )
        )

        unsqueeze_axes_name = self.tensor_name(prefix, "input_unsqueeze_axes")
        self.add_init(graph, unsqueeze_axes_name, np.array([3], dtype=np.int64))
        unsqueezed_input_name = self.tensor_name(prefix, "input_unsqueezed")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [transposed_input_name, unsqueeze_axes_name],
                [unsqueezed_input_name],
                name=self.node_name(prefix, "input_unsqueeze"),
            )
        )

        conv_output_name = self.tensor_name(prefix, "conv_output")
        nodes.append(
            helper.make_node(
                cons.OP_CONV,
                [unsqueezed_input_name, conv_weight_name],
                [conv_output_name],
                name=self.node_name(prefix, "conv"),
                kernel_shape=[1, 1],
            )
        )

        transposed_output_name = self.tensor_name(prefix, "output_transposed")
        nodes.append(
            helper.make_node(
                cons.OP_TRANSPOSE,
                [conv_output_name],
                [transposed_output_name],
                name=self.node_name(prefix, "output_transpose"),
                perm=[0, 2, 1, 3],
            )
        )

        output_shape_name = self.tensor_name(prefix, "output_reshape_shape")
        self.add_init(graph, output_shape_name, np.array([0, 0, -1], dtype=np.int64))
        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [transposed_output_name, output_shape_name],
                [output_name],
                name=self.node_name(prefix, "output_reshape"),
            )
        )
        return nodes

    def _emit_rank4_conv_chain(
        self,
        prefix: str,
        activation_name: str,
        conv_weight_name: str,
        output_name: str,
        output_channels: int,
        input_channels: int,
        activation_shape: list[ShapeValue],
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto] | None:
        head_dim = activation_shape[1]
        sequence_dim = activation_shape[2]

        flattened_shape_name = self.tensor_name(prefix, "flatten_shape")
        flattened_sequence = sequence_dim if isinstance(sequence_dim, int) and sequence_dim > 0 else 0
        self.add_init(
            graph,
            flattened_shape_name,
            np.array([-1, flattened_sequence, input_channels], dtype=np.int64),
        )

        flattened_input_name = self.tensor_name(prefix, "flattened_input")
        nodes: list[onnx.NodeProto] = [
            helper.make_node(
                cons.OP_RESHAPE,
                [activation_name, flattened_shape_name],
                [flattened_input_name],
                name=self.node_name(prefix, "flatten"),
            )
        ]

        inner_output_name = self.tensor_name(prefix, "rank3_output")
        inner_nodes = self._emit_rank3_conv_chain(
            prefix=f"{prefix}_rank3",
            activation_name=flattened_input_name,
            conv_weight_name=conv_weight_name,
            output_name=inner_output_name,
            output_channels=output_channels,
            graph=graph,
        )
        nodes.extend(inner_nodes)

        restored_shape_name = self.tensor_name(prefix, "restore_shape")
        if isinstance(head_dim, int) and isinstance(sequence_dim, int):
            restore_shape = np.array([-1, head_dim, sequence_dim, output_channels], dtype=np.int64)
        else:
            restore_shape = np.array([-1, 0, 0, output_channels], dtype=np.int64)
        self.add_init(graph, restored_shape_name, restore_shape)

        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [inner_output_name, restored_shape_name],
                [output_name],
                name=self.node_name(prefix, "restore"),
            )
        )
        return nodes

    def _rewrite_node(self, node: onnx.NodeProto, graph: onnx.GraphProto) -> None:
        left_name, right_name = node.input[:2]
        output_name = node.output[0]
        prefix = self.get_prefix(node)

        if left_name in self.init_map or right_name in self.init_map:
            plan = self._plan_static_rewrite(prefix, left_name, right_name, output_name)
            if plan is None:
                return
            new_nodes = self._emit_static_rewrite(plan, graph)
        else:
            plan = self._plan_dynamic_rewrite(prefix, left_name, right_name, output_name)
            if plan is None:
                return
            new_nodes = self._emit_dynamic_rewrite(plan, graph)

        if new_nodes is None:
            return

        self.replace_node(node, new_nodes)
        rewritten_conv_names = [candidate.name for candidate in new_nodes if candidate.op_type == cons.OP_CONV]
        if rewritten_conv_names:
            self.log.append(
                f" - MatMul({prefix}) is rewritten as Conv({rewritten_conv_names[0]})"
            )
        else:
            self.log.append(f" - MatMul({prefix}) is rewritten as Mul+ReduceSum")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)
        graph = self.require_graph()

        for node in list(graph.node):
            if node.op_type == cons.OP_MATMUL:
                self._rewrite_node(node, graph)

        self.remove_marked_nodes()
        return model, self.log
