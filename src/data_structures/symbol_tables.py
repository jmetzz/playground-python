from collections.abc import Iterable
from typing import Any


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
