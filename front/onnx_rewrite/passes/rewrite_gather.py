from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import TensorProto, helper

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class GatherNodeContext:
    prefix: str
    axis: int
    data_name: str
    index_name: str
    output_name: str
    data_array: np.ndarray | None
    index_array: np.ndarray | None


class RewriteGather(Folder):
    """Rewrite Gather into a supported expression when the pattern is tractable."""

    def __init__(self) -> None:
        super().__init__()
        self.vocab_cutoff: int | None = None
        self.vocab_cutoff_threshold: int = 5000
        self.chunk_threshold: int = 2000
        self.chunk_size: int = 256

    def _build_context(self, node: onnx.NodeProto) -> GatherNodeContext:
        axis = 0
        for attr in node.attribute:
            if attr.name == "axis":
                axis = int(attr.i)

        data_name, index_name = node.input[:2]
        return GatherNodeContext(
            prefix=self.get_prefix(node),
            axis=axis,
            data_name=data_name,
            index_name=index_name,
            output_name=node.output[0],
            data_array=self.init_map.get(data_name),
            index_array=self.init_map.get(index_name),
        )

    def _rewrite_static_gather(
        self,
        node: onnx.NodeProto,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> bool:
        if context.data_array is None or context.index_array is None:
            return False

        folded = np.take(context.data_array, context.index_array.astype(np.int64), axis=context.axis)
        self.add_init(graph, context.output_name, folded)
        self.mark_for_removal(node)
        self.log.append(f" - Gather({context.prefix}) is rewritten as initializer fold")
        return True

    def _rewrite_scalar_index_gather(
        self,
        node: onnx.NodeProto,
        context: GatherNodeContext,
    ) -> bool:
        if context.index_array is None or context.index_array.size != 1:
            return False

        output_shape = self.shape_info.get(context.output_name)
        if output_shape is None:
            self.log.append(
                f" - Gather({context.prefix}) kept as Gather (output shape unknown for scalar-index path)"
            )
            return True

        index_value = int(context.index_array.item())
        slice_nodes = self._emit_scalar_index_slice(context, output_shape, index_value)
        self.replace_node(node, slice_nodes)
        self.log.append(
            f" - Gather({context.prefix}) is rewritten as Slice+Reshape (axis={context.axis}, index={index_value})"
        )
        return True

    def _rewrite_vocab_gather(
        self,
        node: onnx.NodeProto,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> bool:
        if context.data_array is None or context.axis != 0:
            return False

        vocab_size = context.data_array.shape[0]

        if self.vocab_cutoff is not None and vocab_size > self.vocab_cutoff_threshold:
            new_nodes = self._emit_truncated_vocab_chain(context, graph, self.vocab_cutoff)
            self.replace_node(node, new_nodes)
            self.log.append(
                f" - Gather({context.prefix}) is rewritten as truncated Equal+Where chain (cutoff={min(self.vocab_cutoff, vocab_size)})"
            )
            return True

        if vocab_size > self.chunk_threshold:
            new_nodes = self._emit_chunked_vocab_chain(context, graph)
            self.replace_node(node, new_nodes)
            chunk_count = (vocab_size + self.chunk_size - 1) // self.chunk_size
            self.log.append(
                f" - Gather({context.prefix}) is rewritten as chunked Equal+Mul+ReduceMean ({chunk_count} chunks)"
            )
            return True

        new_nodes = self._emit_small_vocab_chain(context, graph)
        self.replace_node(node, new_nodes)
        self.log.append(
            f" - Gather({context.prefix}) is rewritten as Equal+Where-style accumulation (V={vocab_size})"
        )
        return True

    def _emit_scalar_index_slice(
        self,
        context: GatherNodeContext,
        output_shape: list[int | str],
        index_value: int,
    ) -> list[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []

        starts_name = self.tensor_name(context.prefix, "slice_starts")
        ends_name = self.tensor_name(context.prefix, "slice_ends")
        axes_name = self.tensor_name(context.prefix, "slice_axes")
        self.add_init(self.graph, starts_name, np.array([index_value], dtype=np.int64))
        self.add_init(self.graph, ends_name, np.array([index_value + 1], dtype=np.int64))
        self.add_init(self.graph, axes_name, np.array([context.axis], dtype=np.int64))

        slice_output_name = self.tensor_name(context.prefix, "slice_output")
        nodes.append(
            helper.make_node(
                cons.OP_SLICE,
                [context.data_name, starts_name, ends_name, axes_name],
                [slice_output_name],
                name=self.node_name(context.prefix, "slice"),
            )
        )

        reshape_shape_name = self.tensor_name(context.prefix, "reshape_shape")
        reshape_shape = [dim if isinstance(dim, int) and dim > 0 else 0 for dim in output_shape]
        self.add_init(self.graph, reshape_shape_name, np.array(reshape_shape, dtype=np.int64))
        nodes.append(
            helper.make_node(
                cons.OP_RESHAPE,
                [slice_output_name, reshape_shape_name],
                [context.output_name],
                name=self.node_name(context.prefix, "reshape"),
            )
        )
        return nodes

    def _emit_small_vocab_chain(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        assert context.data_array is not None

        nodes: list[onnx.NodeProto] = []
        unsqueeze_axes_name = self.tensor_name(context.prefix, "mask_unsqueeze_axes")
        self.add_init(graph, unsqueeze_axes_name, np.array([-1], dtype=np.int64))

        running_sum_name: str | None = None
        vocab_size = context.data_array.shape[0]

        for vocab_index in range(vocab_size):
            equality_mask_name = self._emit_index_mask(
                prefix=context.prefix,
                index_name=context.index_name,
                vocab_index=vocab_index,
                graph=graph,
                nodes=nodes,
            )

            unsqueezed_mask_name = self.tensor_name(context.prefix, f"mask_{vocab_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_UNSQUEEZE,
                    [equality_mask_name, unsqueeze_axes_name],
                    [unsqueezed_mask_name],
                    name=self.node_name(context.prefix, f"mask_unsqueeze_{vocab_index}"),
                )
            )

            row_name = self.tensor_name(context.prefix, f"row_{vocab_index}")
            self.add_init(graph, row_name, context.data_array[vocab_index].astype(np.float32))

            weighted_row_name = self.tensor_name(context.prefix, f"weighted_row_{vocab_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_MUL,
                    [unsqueezed_mask_name, row_name],
                    [weighted_row_name],
                    name=self.node_name(context.prefix, f"row_mul_{vocab_index}"),
                )
            )

            if running_sum_name is None:
                running_sum_name = weighted_row_name
                continue

            sum_output_name = (
                context.output_name
                if vocab_index == vocab_size - 1
                else self.tensor_name(context.prefix, f"running_sum_{vocab_index}")
            )
            nodes.append(
                helper.make_node(
                    cons.OP_ADD,
                    [running_sum_name, weighted_row_name],
                    [sum_output_name],
                    name=self.node_name(context.prefix, f"row_add_{vocab_index}"),
                )
            )
            running_sum_name = sum_output_name

        if vocab_size == 1 and running_sum_name != context.output_name:
            passthrough_shape_name = self.tensor_name(context.prefix, "single_vocab_shape")
            self.add_init(graph, passthrough_shape_name, np.array([0, 0], dtype=np.int64))
            nodes.append(
                helper.make_node(
                    cons.OP_RESHAPE,
                    [running_sum_name, passthrough_shape_name],
                    [context.output_name],
                    name=self.node_name(context.prefix, "single_vocab_reshape"),
                )
            )

        return nodes

    def _emit_chunked_vocab_chain(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        assert context.data_array is not None

        vocab_size = context.data_array.shape[0]
        chunk_count = (vocab_size + self.chunk_size - 1) // self.chunk_size
        nodes: list[onnx.NodeProto] = []

        outer_unsqueeze_axes_name = self.tensor_name(context.prefix, "chunk_outer_unsqueeze_axes")
        self.add_init(graph, outer_unsqueeze_axes_name, np.array([-1], dtype=np.int64))
        index_vector_name = self.tensor_name(context.prefix, "chunk_index_vector")
        nodes.append(
            helper.make_node(
                cons.OP_UNSQUEEZE,
                [context.index_name, outer_unsqueeze_axes_name],
                [index_vector_name],
                name=self.node_name(context.prefix, "chunk_index_unsqueeze"),
            )
        )

        inner_unsqueeze_axes_name = self.tensor_name(context.prefix, "chunk_inner_unsqueeze_axes")
        self.add_init(graph, inner_unsqueeze_axes_name, np.array([-1], dtype=np.int64))

        running_sum_name: str | None = None
        for chunk_index in range(chunk_count):
            start = chunk_index * self.chunk_size
            end = min(start + self.chunk_size, vocab_size)
            actual_chunk_size = end - start

            range_name = self.tensor_name(context.prefix, f"chunk_range_{chunk_index}")
            self.add_init(graph, range_name, np.arange(start, end, dtype=np.int64))

            equality_name = self.tensor_name(context.prefix, f"chunk_equal_{chunk_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_EQUAL,
                    [index_vector_name, range_name],
                    [equality_name],
                    name=self.node_name(context.prefix, f"chunk_equal_{chunk_index}"),
                )
            )

            float_mask_name = self.tensor_name(context.prefix, f"chunk_mask_{chunk_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_CAST,
                    [equality_name],
                    [float_mask_name],
                    name=self.node_name(context.prefix, f"chunk_cast_{chunk_index}"),
                    to=TensorProto.FLOAT,
                )
            )

            expanded_mask_name = self.tensor_name(context.prefix, f"chunk_mask_expanded_{chunk_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_UNSQUEEZE,
                    [float_mask_name, inner_unsqueeze_axes_name],
                    [expanded_mask_name],
                    name=self.node_name(context.prefix, f"chunk_unsqueeze_{chunk_index}"),
                )
            )

            chunk_weight_name = self.tensor_name(context.prefix, f"chunk_weight_{chunk_index}")
            scaled_chunk = context.data_array[start:end].astype(np.float32) * float(actual_chunk_size)
            self.add_init(graph, chunk_weight_name, scaled_chunk)

            weighted_chunk_name = self.tensor_name(context.prefix, f"chunk_weighted_{chunk_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_MUL,
                    [expanded_mask_name, chunk_weight_name],
                    [weighted_chunk_name],
                    name=self.node_name(context.prefix, f"chunk_mul_{chunk_index}"),
                )
            )

            reduced_chunk_name = self.tensor_name(context.prefix, f"chunk_reduced_{chunk_index}")
            nodes.append(
                helper.make_node(
                    cons.OP_REDUCE_MEAN,
                    [weighted_chunk_name],
                    [reduced_chunk_name],
                    name=self.node_name(context.prefix, f"chunk_reduce_mean_{chunk_index}"),
                    axes=[-2],
                    keepdims=0,
                )
            )

            if running_sum_name is None:
                running_sum_name = reduced_chunk_name
                continue

            sum_output_name = (
                context.output_name
                if chunk_index == chunk_count - 1
                else self.tensor_name(context.prefix, f"chunk_sum_{chunk_index}")
            )
            nodes.append(
                helper.make_node(
                    cons.OP_ADD,
                    [running_sum_name, reduced_chunk_name],
                    [sum_output_name],
                    name=self.node_name(context.prefix, f"chunk_add_{chunk_index}"),
                )
            )
            running_sum_name = sum_output_name

        if chunk_count == 1 and running_sum_name != context.output_name:
            passthrough_shape_name = self.tensor_name(context.prefix, "chunk_single_shape")
            self.add_init(graph, passthrough_shape_name, np.array([0], dtype=np.int64))
            nodes.append(
                helper.make_node(
                    cons.OP_RESHAPE,
                    [running_sum_name, passthrough_shape_name],
                    [context.output_name],
                    name=self.node_name(context.prefix, "chunk_single_reshape"),
                )
            )

        return nodes

    def _emit_truncated_vocab_chain(
        self,
        context: GatherNodeContext,
        graph: onnx.GraphProto,
        cutoff: int,
    ) -> list[onnx.NodeProto]:
        assert context.data_array is not None
        truncated_context = GatherNodeContext(
            prefix=context.prefix,
            axis=context.axis,
            data_name=context.data_name,
            index_name=context.index_name,
            output_name=context.output_name,
            data_array=context.data_array[: min(cutoff, context.data_array.shape[0])],
            index_array=context.index_array,
        )
        return self._emit_small_vocab_chain(truncated_context, graph)

    def _emit_index_mask(
        self,
        prefix: str,
        index_name: str,
        vocab_index: int,
        graph: onnx.GraphProto,
        nodes: list[onnx.NodeProto],
    ) -> str:
        scalar_index_name = self.tensor_name(prefix, f"index_{vocab_index}")
        self.add_init(graph, scalar_index_name, np.array(vocab_index, dtype=np.int64))

        equality_name = self.tensor_name(prefix, f"equal_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_EQUAL,
                [index_name, scalar_index_name],
                [equality_name],
                name=self.node_name(prefix, f"equal_{vocab_index}"),
            )
        )

        float_mask_name = self.tensor_name(prefix, f"float_mask_{vocab_index}")
        nodes.append(
            helper.make_node(
                cons.OP_CAST,
                [equality_name],
                [float_mask_name],
                name=self.node_name(prefix, f"mask_cast_{vocab_index}"),
                to=TensorProto.FLOAT,
            )
        )
        return float_mask_name

    def _rewrite_node(self, node: onnx.NodeProto, graph: onnx.GraphProto) -> None:
        context = self._build_context(node)

        if self._rewrite_static_gather(node, context, graph):
            return
        if self._rewrite_scalar_index_gather(node, context):
            return
        if self._rewrite_vocab_gather(node, context, graph):
            return

        self.log.append(f" - Gather({context.prefix}) kept as Gather (dynamic and unsupported)")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)
        graph = self.require_graph()

        for node in list(graph.node):
            if node.op_type == cons.OP_GATHER:
                self._rewrite_node(node, graph)

        self.remove_marked_nodes()
        return model, self.log
