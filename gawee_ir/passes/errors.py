from __future__ import annotations
from typing     import List, Any 

basic = "PASS-PART ERROR"

class NotImplementedError(Exception):
    def __init__(
        self,
        category: str, 
        operator: str, 
    ):
        self.category = category
        self.operator = operator
        self.msg = basic 

        msg = [self.msg, self.category, self.operator]

        super().__init__(" | ".join(msg))
        return 


class PythonOpError(Exception):
    def __init__(
        self, 
        category: str, 
        data: Any
    ):
        self.category = category
        self.data     = data 
        self.msg      = "Python operator errors" 

        msg = [self.msg, self.category, self.data]
        super().__init__(" | ".join(msg))
        return 