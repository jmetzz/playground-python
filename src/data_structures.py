from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Self

import numpy as np


class BinaryTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def deserialize(arr: List[Optional[int]], root: int = 0):
        """
        Recursively constructs the tree based on the array representation,
        handling None values to skip creating nodes for missing children.

        Args:
            arr:
            root:

        Returns:

        """
        if root >= len(arr) or arr[root] is None:
            return None

        return BinaryTreeNode(
            arr[root],
            BinaryTreeNode.deserialize(arr, root * 2 + 1),
            BinaryTreeNode.deserialize(arr, root * 2 + 2),
        )

    @staticmethod
    def from_indices(indices: List[int]):
        size = max(indices) + 1
        arr = np.full(size, None)
        arr[indices] = indices
        return BinaryTreeNode.deserialize(arr.tolist())

    @staticmethod
    def from_map(mapping: Dict[int, int]):
        raise NotImplementedError

    @classmethod
    def preorder(cls, r):
        if not r:  # base case
            return None

        # traverse the subtrees
        left = cls.preorder(r.left) or []
        right = cls.preorder(r.right) or []
        return [r.val] + left + right

    @classmethod
    def inorder(cls, r):
        if not r:  # base case
            return None
        # traverse the subtrees
        left = cls.inorder(r.left) or []
        right = cls.inorder(r.right) or []

        # visit the current node and collect the elements in left, in appropriate order
        return left + [r.val] + right

    @classmethod
    def postorder(cls, r):
        if not r:
            return None
        left = cls.postorder(r.left) or []
        right = cls.postorder(r.right) or []
        # visit the node and collect its value
        return left + right + [r.val]

    def bfs_apply(self, func):
        q = deque()
        q.append(self)

        while q:
            curr_node = q.popleft()
            if not curr_node:
                continue
            # visit function
            func(curr_node)
            q.append(curr_node.left)
            q.append(curr_node.right)

    def bfs(self):
        q = deque()
        q.append(self)
        elements = []
        while q:
            curr_node = q.popleft()
            if not curr_node:
                continue
            elements.append(curr_node.val)
            q.append(curr_node.left)
            q.append(curr_node.right)
        return elements

    def to_map(self) -> Dict[int, int]:
        raise NotImplementedError

    def serialize(self, prefix: str = "", is_left=True) -> str:
        """Generate a string representation of a binary tree"""
        # if node is None:
        #     return "Null\n"
        result = prefix
        if prefix == "":
            result += "── "
        else:
            result += "├── " if is_left else "└── "

        result += str(self.val) + "\n"
        if self.left is not None:
            result += self.left.serialize(prefix + ("   " if not is_left or self.right is not None else "    "), True)

        if self.right is not None:
            result += self.right.serialize(prefix + ("   " if not is_left or self.left is not None else "    "), False)

        return result

    def invert(self):
        q = deque()
        q.append(self)
        while q:
            node = q.popleft()
            if not node:
                continue
            node.left, node.right = node.right, node.left
            q.append(node.left)
            q.append(node.right)
        return self

    @classmethod
    def invert_recursive(cls, root):
        if not root:
            return None
        root.left, root.right = root.right, root.left
        cls.invert_recursive(root.left)
        cls.invert_recursive(root.right)
        return root

    @classmethod
    def is_equal(cls, one: Self, other: Self) -> bool:
        if one is None and one is None:
            return True

        if one is not None and other is not None and one.val == other.val:
            return cls.is_equal(one.left, other.left) and cls.is_equal(one.right, other.right)
        else:
            return False


class SingleLinkListNode:
    def __init__(self, val: Any = None, next_node: Self = None):
        self.val = val
        self.next = next_node

    @classmethod
    def from_array(cls, arr: List[Any]) -> Self:
        if not arr:
            return None

        root = SingleLinkListNode(arr[0])
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


class UnionFind:
    def __init__(self, nodes: int):
        # Initialize parent and rank arrays.
        # Each node is its own parent
        self.parent = [i for i in range(nodes)]
        self.rank = [0] * nodes
        self.components_count = nodes

    def find(self, x: int) -> int:
        # Find the parent of node x. Use Path Compression
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def unite(self, a: int, b: int) -> None:
        # Unite two nodes x and y, if they are not already united
        p_a = self.find(a)
        p_b = self.find(b)
        if p_a != p_b:
            # Union by Rank Heuristic
            if self.rank[p_a] < self.rank[p_b]:
                self.parent[p_a] = p_b
                self.rank[p_b] += self.rank[p_a]
            else:
                self.parent[p_b] = p_a
                self.rank[p_a] += self.rank[p_b]
            self.components_count -= 1

    def connected(self, x: int, y: int) -> bool:
        # Check if two nodes x and y are connected or not
        return self.find(x) == self.find(y)

    def reset(self, x: int) -> None:
        # Reset the initial properties of node x
        self.parent[x] = x
        self.rank[x] = 0
        self.components_count += 1


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


class TrieSymbolTable:
    class TrieNode:
        def __init__(self, val: Any = None, next_node: List[Self] = None):
            self.val = val
            self.next = next_node

    def __init__(self):
        self._R = 256
        self._root = TrieSymbolTable.TrieNode(None, [TrieSymbolTable.TrieNode()] * self._R)

    def get(self, key: str) -> Any:
        node = self._get(self._root, key, 0)
        return node.val if node else None

    def _get(self, node: TrieNode, key: str, depth: int) -> Any:
        if not node or depth == len(key):
            return node
        sub_tree_idx = ord(key[depth])
        return self._get(node.next[sub_tree_idx], key, depth + 1)

    def put(self, key: str, value: Any) -> None:
        self._root = self._put(self._root, key, value, 0)

    def _put(self, node: TrieNode, key: str, value: Any, depth: int) -> TrieNode:
        if node is None:
            node = TrieSymbolTable.TrieNode()
        if depth == len(key):
            node.val = value
            return node

        sub_tree_idx = ord(key[depth])
        node.next[sub_tree_idx] = self._put(node.next[sub_tree_idx], key, value, depth + 1)
        return node


if __name__ == "__main__":
    tree = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    accumulator = []

    def swap_children(node):
        if not node:
            return
        node.left, node.right = node.right, node.left

    tree.bfs_apply(lambda x: accumulator.append(x.val))
    print(f"tree elements (bfs): {accumulator}")
    tree.bfs_apply(func=swap_children)
    accumulator = []
    tree.bfs_apply(lambda x: accumulator.append(x.val))
    print(f"tree elements (swap): {accumulator}")

    inverted = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9]).invert().bfs()
    print(f"tree elements (invert): {inverted}")

    t2 = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    inverted = BinaryTreeNode.invert_recursive(t2).bfs()
    print(f"tree elements (invert_recursive): {inverted}")
