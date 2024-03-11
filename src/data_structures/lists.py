from typing import Any, List, Self


class SingleLinkNode:
    def __init__(self, val: Any = None, next_node: Self = None):
        self.val = val
        self.next = next_node

    @classmethod
    def from_array(cls, arr: List[Any]) -> Self:
        if not arr:
            return None

        root = SingleLinkNode(arr[0])
        root.next = cls.from_array(arr[1:])
        return root

    def serialize(self) -> str:
        return str(self.asarray())

    def asarray(self) -> List[Any]:
        arr = []
        visitor = self
        while visitor:
            arr.append(visitor.val)
            visitor = visitor.next
        return arr
