from __future__ import annotations

import onnx

from .cleanup import Cleanup
from .constant_folding import ConstantFolding
from .eliminate_id import EliminateId
from .rewrite_bn import RewriteBN
from .rewrite_gather import RewriteGather
from .rewrite_gemm import RewriteGemm
from .rewrite_matmul import RewriteMatmul
from .rewrite_pow import RewritePow


class Passer:
    """Run the frontend rewrite pipeline in a fixed order."""

    def __init__(self) -> None:
        self.passes = [
            ConstantFolding(),
            EliminateId(),
            RewriteBN(),
            RewritePow(),
            RewriteGather(),
            RewriteGemm(),
            RewriteMatmul(),
            Cleanup(),
        ]

    def optimize(self, model: onnx.ModelProto) -> tuple[onnx.ModelProto, list[str]]:
        all_logs: list[str] = []

        for rewrite_pass in self.passes:
            pass_name = rewrite_pass.__class__.__name__
            before_nodes = len(model.graph.node)
            before_initializers = len(model.graph.initializer)
            model, logs = rewrite_pass.run(model)
            after_nodes = len(model.graph.node)
            after_initializers = len(model.graph.initializer)
            all_logs.append(
                f"PASS {pass_name}: "
                f"nodes {before_nodes} -> {after_nodes}, "
                f"initializers {before_initializers} -> {after_initializers}, "
                f"removed {rewrite_pass.deleted_node}"
            )
            all_logs.extend(f"  {line}" for line in logs)

        return model, all_logs
