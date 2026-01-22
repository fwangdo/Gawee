from __future__ import annotations
from typing import List 


class DimensionError(Exception):
    def __init__(
        self,
        op: str,
        expected: str | List[int] | None = None,
        actual: List[int] | None = None,
        msg: str | None = None,
    ):
        self.op = op
        self.expected = expected
        self.actual = actual

        detail = []
        detail.append(f"operator={op}")

        if expected is not None:
            detail.append(f"expected={expected}")
        if actual is not None:
            detail.append(f"actual={list(actual)}")
        if msg is not None:
            detail.append(msg)

        super().__init__(" | ".join(detail))
        return 


class NoneCaseError(Exception):
    
    def __init__(
        self,
        op,
        is_input: bool, 
        is_output: bool, 
    ):     
        self.op = op
        self.is_input = is_input
        self.is_output = is_output

        detail = [self.op, self.is_input, self.is_output]

        super().__init__(" | ".join(detail))
        return 