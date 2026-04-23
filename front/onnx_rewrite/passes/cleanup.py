from __future__ import annotations

from collections import deque
from typing import Deque

import onnx

from .folder import Folder


class Cleanup(Folder):
    """
    Minimal cleanup pass.

    V1 only runs ONNX checker and leaves topology untouched.
    This pass exists to reserve the final normalization/validation slot in the
    pipeline before real graph rewrites are added.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _remove_unused_initializers(graph: onnx.GraphProto) -> int:
        """Remove initializers not referenced by any node or graph input.

        Args:
            graph: ONNX graph to clean up.

        Returns:
            Number of removed initializers.
        """
        used_names: set[str] = set()
        for node in graph.node:
            for inp in node.input:
                if inp:
                    used_names.add(inp)
        for inp in graph.input:
            used_names.add(inp.name)

        unused = [init for init in graph.initializer if init.name not in used_names]
        for init in unused:
            graph.initializer.remove(init)

        return len(unused)

    @staticmethod
    def _topological_sort(graph: onnx.GraphProto) -> None:
        """Topologically sort graph nodes using Kahn's algorithm.

        Ensures every node appears after its dependencies (producers).
        Falls back to appending remaining nodes if cycles or
        disconnected subgraphs exist.

        Args:
            graph: ONNX graph whose nodes will be reordered in-place.
        """
        nodes = list(graph.node)
        available_inputs = {value.name for value in graph.input}
        available_initializers = {initializer.name for initializer in graph.initializer}
        available_values = available_inputs | available_initializers

        output_to_node: dict[str, int] = {}
        for index, node in enumerate(nodes):
            for output_name in node.output:
                if output_name:
                    output_to_node[output_name] = index

        consumers_by_node: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
        in_degree: list[int] = [0] * len(nodes)

        for consumer_index, node in enumerate(nodes):
            producer_indices: set[int] = set()
            for input_name in node.input:
                if not input_name or input_name in available_values:
                    continue
                producer_index = output_to_node.get(input_name)
                if producer_index is None or producer_index == consumer_index:
                    continue
                producer_indices.add(producer_index)

            in_degree[consumer_index] = len(producer_indices)
            for producer_index in producer_indices:
                consumers_by_node[producer_index].append(consumer_index)

        queue: Deque[int] = deque(
            index for index, dependency_count in enumerate(in_degree) if dependency_count == 0
        )
        sorted_indices: list[int] = []

        while queue:
            node_index = queue.popleft()
            sorted_indices.append(node_index)
            for consumer_index in consumers_by_node[node_index]:
                in_degree[consumer_index] -= 1
                if in_degree[consumer_index] == 0:
                    queue.append(consumer_index)

        seen = set(sorted_indices)
        if len(sorted_indices) != len(nodes):
            sorted_indices.extend(index for index in range(len(nodes)) if index not in seen)

        del graph.node[:]
        for node_index in sorted_indices:
            graph.node.append(nodes[node_index])

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        """Clean up the model: remove unused initializers, sort, validate.

        Args:
            model: ONNX model to clean up.

        Returns:
            (model, log): Cleaned model and list of log messages.
        """
        self.deleted_node = 0
        graph = model.graph
        log: list[str] = []

        removed = self._remove_unused_initializers(graph)
        if removed:
            log.append(f"[Cleanup] removed {removed} unused initializers")

        self._topological_sort(graph)
        log.append("[Cleanup] topologically sorted graph nodes")

        try:
            onnx.checker.check_model(model)
            log.append("[Cleanup] onnx.checker.check_model passed")
        except Exception as exc:
            log.append(f"[Cleanup] onnx.checker.check_model failed: {exc}")
            raise
        return model, log
