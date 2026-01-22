from __future__ import annotations
from typing import List 

basic = "PASS-PART ERROR"

class NotImplementedError(Exception):
    def __init__(
        self,
        category: str, 
        operator: str, 
    ):
        self.category = category
        self.operator = operator

        msg = [basic, self.category, self.operator]

        super().__init__(" | ".join(msg))
        return 