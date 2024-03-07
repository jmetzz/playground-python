from typing import Any, Iterable, List, Self


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


class SymbolTable:
    def __init__(self):
        pass

    def put(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    def get(self, key: Any) -> Any:
        raise NotImplementedError

    def delete(self, key: Any) -> None:
        raise NotImplementedError

    def contains(self, key: Any) -> bool:
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def keys(self) -> Iterable[Any]:
        raise NotImplementedError


class OrderedSymbolTable:
    def __init__(self):
        pass

    def put(self, key: Any, value: Any) -> None:
        raise NotImplementedError

    def get(self, key: Any) -> Any:
        raise NotImplementedError

    def delete(self, key: Any) -> None:
        raise NotImplementedError

    def contains(self, key: Any) -> bool:
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def keys(self) -> Iterable[Any]:
        raise NotImplementedError

    def range_keys(self, low_key: Any, high_key: Any) -> Iterable[Any]:
        raise NotImplementedError

    def min(self) -> Any:
        raise NotImplementedError

    def max(self) -> Any:
        raise NotImplementedError

    def floor(self, key: Any) -> Any:
        raise NotImplementedError

    def ceiling(self, key: Any) -> Any:
        raise NotImplementedError

    def rank(self, key: Any) -> int:
        raise NotImplementedError

    def select(self, k: int) -> Iterable[Any]:
        """Return the key of rank k"""
        raise NotImplementedError

    def delete_min(self) -> None:
        raise NotImplementedError

    def delete_max(self) -> None:
        raise NotImplementedError


class StringSymbolTable:
    def __init__(self):
        pass

    def put(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def get(self, key: str) -> Any:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def contains(self, key: str) -> bool:
        raise NotImplementedError

    def empty(self) -> bool:
        raise NotImplementedError

    def longest_prefix_of(self, sequence: str) -> str:
        raise NotImplementedError

    def matching_keys(self, value: str) -> Iterable[str]:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError

    def keys(self) -> Iterable[str]:
        raise NotImplementedError
