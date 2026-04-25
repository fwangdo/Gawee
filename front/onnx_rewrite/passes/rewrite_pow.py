from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper

from ..utils import cons
from .folder import Folder


@dataclass(frozen=True)
class PowNodeContext:
    prefix: str
    base_name: str
    exponent_name: str
    output_name: str
    exponent: float | None


class RewritePow(Folder):
    """Rewrite Pow with scalar constant exponent into supported ops."""

    def _build_context(self, node: onnx.NodeProto) -> PowNodeContext:
        base_name, exponent_name = node.input[:2]
        exponent = self._get_scalar_exponent(exponent_name)
        return PowNodeContext(
            prefix=self.get_prefix(node),
            base_name=base_name,
            exponent_name=exponent_name,
            output_name=node.output[0],
            exponent=exponent,
        )

    def _get_scalar_exponent(self, exponent_name: str) -> float | None:
        exponent = self.init_map.get(exponent_name)
        if exponent is None or exponent.size != 1:
            return None
        return float(exponent.reshape(-1)[0])

    @staticmethod
    def _is_close(left: float, right: float) -> bool:
        return abs(left - right) <= 1e-6

    def _redirect_value(self, old_name: str, new_name: str) -> None:
        graph = self.require_graph()

        for node in graph.node:
            if node.op_type == cons.OP_POW:
                continue
            for index, input_name in enumerate(node.input):
                if input_name == old_name:
                    node.input[index] = new_name

        for output in graph.output:
            if output.name == old_name:
                output.name = new_name

    def _make_mul_chain(
        self,
        context: PowNodeContext,
        power: int,
    ) -> list[onnx.NodeProto]:
        nodes: list[onnx.NodeProto] = []
        running_name = context.base_name

        for index in range(2, power + 1):
            output_name = (
                context.output_name
                if index == power
                else self.tensor_name(context.prefix, f"pow_{index}")
            )
            nodes.append(
                helper.make_node(
                    cons.OP_MUL,
                    [running_name, context.base_name],
                    [output_name],
                    name=self.node_name(context.prefix, f"mul_{index}"),
                )
            )
            running_name = output_name

        return nodes

    def _make_reciprocal(
        self,
        context: PowNodeContext,
        input_name: str,
    ) -> list[onnx.NodeProto]:
        one_name = self.tensor_name(context.prefix, "one")
        self.add_init(self.graph, one_name, np.array(1.0, dtype=np.float32))
        return [
            helper.make_node(
                cons.OP_DIV,
                [one_name, input_name],
                [context.output_name],
                name=self.node_name(context.prefix, "reciprocal"),
            )
        ]

    def _make_sqrt(self, context: PowNodeContext, output_name: str) -> onnx.NodeProto:
        return helper.make_node(
            cons.OP_SQRT,
            [context.base_name],
            [output_name],
            name=self.node_name(context.prefix, "sqrt"),
        )

    def _make_rewrite_nodes(
        self,
        context: PowNodeContext,
    ) -> list[onnx.NodeProto] | None:
        exponent = context.exponent
        if exponent is None:
            self.log.append(f" - Pow({context.prefix}) kept as Pow (dynamic exponent)")
            return None

        if self._is_close(exponent, 1.0):
            self._redirect_value(context.output_name, context.base_name)
            self.log.append(f" - Pow({context.prefix}) is removed (exponent=1)")
            return []

        if self._is_close(exponent, 0.5):
            return [self._make_sqrt(context, context.output_name)]

        if self._is_close(exponent, -0.5):
            sqrt_name = self.tensor_name(context.prefix, "sqrt")
            return [
                self._make_sqrt(context, sqrt_name),
                *self._make_reciprocal(context, sqrt_name),
            ]

        if self._is_close(exponent, -1.0):
            return self._make_reciprocal(context, context.base_name)

        if exponent > 1.0 and self._is_close(exponent, round(exponent)):
            power = int(round(exponent))
            if 2 <= power <= 4:
                return self._make_mul_chain(context, power)

        self.log.append(f" - Pow({context.prefix}) kept as Pow (unsupported exponent={exponent:g})")
        return None

    def _rewrite_node(self, node: onnx.NodeProto) -> None:
        context = self._build_context(node)
        new_nodes = self._make_rewrite_nodes(context)
        if new_nodes is None:
            return

        self.replace_node(node, new_nodes)
        if new_nodes:
            op_names = "+".join(node.op_type for node in new_nodes)
            self.log.append(f" - Pow({context.prefix}) is rewritten as {op_names}")

    def run(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        self.prepare(model)

        for node in list(self.graph.node):
            if node.op_type == cons.OP_POW:
                self._rewrite_node(node)

        self.remove_marked_nodes()
        return model, self.log
