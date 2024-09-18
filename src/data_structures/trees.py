import bisect
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Self

import numpy as np


class BinaryTreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def deserialize(arr: list[int | None], root: int = 0) -> Self:
        """Recursively constructs the tree based on the array representation,
        handling None values to skip creating nodes for missing children.

        Args:
        ----
            arr:
            root:

        Returns:
        -------

        """
        if root >= len(arr) or arr[root] is None:
            return None

        return BinaryTreeNode(
            arr[root],
            BinaryTreeNode.deserialize(arr, root * 2 + 1),
            BinaryTreeNode.deserialize(arr, root * 2 + 2),
        )

    def serialize(self) -> list[int | None]:
        """Serializes the tree to an array representation using level-order traversal."""
        if not self:
            return []

        result = []
        queue = deque([self])

        while queue:
            node = queue.popleft()
            if node:
                result.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                result.append(None)

        # Remove trailing None values to minimize the array length
        while result and result[-1] is None:
            result.pop()

        return result

    def serialize_tree(self, prefix: str = "", is_left=True) -> str:
        """Generate a string representation of a binary tree"""

        tree_str = prefix
        if prefix == "":
            tree_str += "── "
        else:
            tree_str += "├── " if is_left else "└── "

        tree_str += str(self.val) + "\n"
        if self.left is not None:
            tree_str += self.left.serialize_tree(
                prefix + ("   " if not is_left or self.right is not None else "    "), True
            )

        if self.right is not None:
            tree_str += self.right.serialize_tree(
                prefix + ("   " if not is_left or self.left is not None else "    "), False
            )

        return tree_str

    @staticmethod
    def from_indices(indices: List[int]):
        size = max(indices) + 1
        arr = np.full(size, None)
        arr[indices] = indices
        return BinaryTreeNode.deserialize(arr.tolist())

    @classmethod
    def from_array(cls, values: list[int]) -> Self:
        """Function to construct a binary tree from a list (level-order)"""
        if not values:
            return None

        root = cls(values[0])
        queue = [root]
        i = 1

        while queue and i < len(values):
            node = queue.pop(0)

            if i < len(values) and values[i] is not None:
                node.left = cls(values[i])
                queue.append(node.left)
            i += 1

            if i < len(values) and values[i] is not None:
                node.right = cls(values[i])
                queue.append(node.right)
            i += 1

        return root

    @classmethod
    def find_node_by_value(cls, root: Self, val: int):
        """Helper function to find the reference to a node by its value"""
        if not root:
            return None
        if root.val == val:
            return root

        left_search = cls.find_node_by_value(root.left, val)
        if left_search:
            return left_search

        return cls.find_node_by_value(root.right, val)

    @staticmethod
    def from_map(mapping: dict[int, int]):
        raise NotImplementedError

    @classmethod
    def preorder(cls, root: Self) -> list[int]:
        if not root:  # base case
            return None

        # traverse the subtrees
        left = cls.preorder(root.left) or []
        right = cls.preorder(root.right) or []
        return [root.val] + left + right

    @classmethod
    def inorder(cls, root: Self) -> list[int]:
        if not root:  # base case
            return None
        # traverse the subtrees
        left = cls.inorder(root.left) or []
        right = cls.inorder(root.right) or []

        # visit the current node and collect the elements in left, in appropriate order
        return left + [root.val] + right

    @classmethod
    def postorder(cls, root: Self) -> list[int]:
        if not root:
            return None
        left = cls.postorder(root.left) or []
        right = cls.postorder(root.right) or []
        # visit the node and collect its value
        return left + right + [root.val]

    @staticmethod
    def level_order(root: Self | None) -> list[list[int]]:
        if not root:
            return []

        queue = deque([root])
        answer = []

        while queue:
            level_size, level_values = len(queue), []
            for _ in range(level_size):
                curr = queue.popleft()
                level_values.append(curr.val)
                if curr.left:
                    queue.append(curr.left)
                if curr.right:
                    queue.append(curr.right)
            answer.append(level_values)

        return answer

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
        return False

    def is_symmetric(self) -> bool:
        return self.are_symmetric(self.left, self.right)

    @classmethod
    def are_symmetric(cls, root1, root2) -> bool:
        if root1 is None and root2 is None:
            return True
        if (root1 is None) != (root2 is None) or root1.val != root2.val:
            return False
        # both root nodes are valid nodes
        return cls.are_symmetric(root1.left, root2.right) and cls.are_symmetric(root1.right, root2.left)

    def __str__(self):
        return " -> ".join(self.inorder(root=self))


class TrieSymbolTableRecursive:
    class TrieNode:
        def __init__(self, val: Any = None, next_node: List[Optional[Self]] = None):
            self.val = val
            self.next = next_node

    def __init__(self, r: int = 256):
        self._R = r
        self._root = TrieSymbolTableRecursive.TrieNode(None, [None] * r)
        self._size = 0

    def get(self, key: str) -> Any:
        target = self._get(self._root, key, 0)
        return target.val if target else None

    @classmethod
    def _get(cls, node: TrieNode, key: str, curr_idx: int) -> Any:
        if not node or curr_idx == len(key):
            return node
        sub_tree_idx = ord(key[curr_idx])
        return cls._get(node.next[sub_tree_idx], key, curr_idx + 1)

    def insert(self, key: str, value: Any) -> None:
        def _put(node: TrieSymbolTableRecursive.TrieNode, curr_idx: int) -> TrieSymbolTableRecursive.TrieNode:
            if node is None:
                node = TrieSymbolTableRecursive.TrieNode(None, [None] * self._R)
            if curr_idx == len(key):
                node.val = value
                return node

            sub_tree_idx = ord(key[curr_idx])
            node.next[sub_tree_idx] = _put(node.next[sub_tree_idx], curr_idx + 1)
            return node

        self._root = _put(self._root, 0)
        self._size += 1

    def size(self) -> int:
        return self._size

    def lazy_size(self) -> int:
        """Calculates the number of keys in the Trie recursively

        This function is for learning purposes and should not be used for real applications,
        since it might be too slow for clients.
        """

        def _lazy_size(node) -> int:
            if node is None:
                return 0
            counter = 1 if node.val else 0
            for sub_tree in range(self._R):
                counter += _lazy_size(node.next[sub_tree])
            return counter

        return _lazy_size(self._root)

    def keys(self) -> Iterable[str]:
        return self.keys_with_prefix("")

    def keys_with_prefix(self, prefix: str) -> Iterable[str]:
        """Fetches all keys starting with the given prefix"""
        start_node = self._get(self._root, prefix, 0)  # the root node with this prefix
        queue = []
        self._collect(start_node, prefix, queue, first_only=False)  # all keys from this node onwards
        return queue

    def is_valid_prefix(self, prefix: str):
        """Checks if there any at least on key starting with prefix"""
        start_node = self._get(self._root, prefix, 0)  # the root node with this prefix
        queue = []
        self._collect(start_node, prefix, queue, first_only=True)  # all keys from this node onwards
        return len(queue) > 0

    def _collect(self, node: TrieNode, prefix: str, accumulating_queue: List[str], first_only: bool = False):
        if node is None:
            return
        if node.val:
            # found a key
            accumulating_queue.append(prefix)
            if first_only:
                return
        for idx in range(self._R):
            self._collect(node.next[idx], prefix + chr(idx), accumulating_queue, first_only)

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def delete(self, key: str) -> None:
        self._root = self._del(self._root, key, 0)

    def _del(self, node: TrieNode, key: str, curr_idx: int) -> Optional[TrieNode]:
        if node is None or curr_idx > len(key):
            # here we should log a warning message
            # saying the client is trying to remove a non-existing node
            return None
        if curr_idx == len(key) and node.val:
            node.val = None  # removing the key
            self._size -= 1
        else:
            # recursively follows the next subtree
            sub_tree_idx = ord(key[curr_idx])
            node.next[sub_tree_idx] = self._del(node.next[sub_tree_idx], key, curr_idx + 1)

        # remove empty subtrees if necessary
        return node if node.val or any(node.next) else None

    def empty(self) -> bool:
        return self._size == 0

    def longest_prefix_of(self, sequence: str) -> str:
        raise NotImplementedError

    def matching_keys(self, value: str) -> Iterable[str]:
        raise NotImplementedError


class TrieSymbolTableDict:
    """ab bc be

       /\
      a  b
     b  c e

    TrieNode = dict[char, TrieNode]

    root = {
        "a": {
                "b": {
                    "EOW": None
                }
        }
        "b": {
            "c": {"EOW": None},
            "e": {"EOW": None},
            "EOW": None,
        }
    }
    """

    EOW_TOKEN = "#"

    def __init__(self):
        self._root = {}
        self._size = 0

    def insert(self, key: str) -> None:
        node = self._root
        for letter in key:
            if letter not in node:
                node[letter] = {}
            node = node[letter]
        node[self.EOW_TOKEN] = None
        self._size += 1

    def get(self, key: str) -> Dict[str, Dict]:
        node = self._get_subtree(self._root, key)
        return node if node and self.EOW_TOKEN in node else None

    @classmethod
    def _get_subtree(cls, node, key: str) -> Dict[str, Dict]:
        for letter in key:
            if letter not in node:
                return None
            node = node[letter]
        return node

    def size(self) -> int:
        return self._size

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def is_valid_prefix(self, prefix: str):
        node = self._root

        for letter in prefix:
            if letter not in node:
                return False
            node = node[letter]
        return True

    def keys(self) -> Iterable[str]:
        return self._keys(self._root)

    @classmethod
    def _keys(cls, node) -> Iterable[str]:
        container = []

        def dfs(node, prefix_so_far):
            if not node:
                return

            if cls.EOW_TOKEN in node:
                container.append(prefix_so_far)

            for letter, sub_tree in node.items():
                dfs(sub_tree, prefix_so_far + letter)

        dfs(node, "")
        return container

    def keys_with_prefix(self, prefix: str) -> Iterable[str]:
        node = self._get_subtree(self._root, prefix)  # the root of the tree starting at prefix

        # now, collect all keys from this node onwards
        suffixes = self._keys(node)
        if suffixes:
            return [prefix + s for s in suffixes]

        return []

    def delete(self, key: str) -> None:
        node = self._root
        for letter in key:
            if letter in node:
                node = node[letter]
        if node and self.EOW_TOKEN in node:
            node.pop(self.EOW_TOKEN)
            self._size -= 1

    def empty(self) -> bool:
        return self._size == 0

    def longest_prefix_of(self, sequence: str) -> str:
        raise NotImplementedError

    def matching_keys(self, value: str) -> Iterable[str]:
        raise NotImplementedError

    def longest_common_prefix(
        self,
    ) -> str:
        prefix = ""
        node = self._root

        while node and self.EOW_TOKEN not in node and len(node) == 1:
            # get the first (and presumably only) key-value pair from the current node
            letter, node = next(iter(node.keys()))
            prefix += letter

        return prefix


class TrieSymbolTableBisect:
    def __init__(self):
        self._trie = []

    def insert(self, word: str) -> None:
        bisect.insort(self._trie, word)

    def get(self, word: str) -> bool:
        i = bisect.bisect(self._trie, word)
        return i > 0 and self._trie[i - 1] == word

    def is_valid_prefix(self, prefix: str) -> bool:
        j = bisect.bisect_left(self._trie, prefix)
        return j < len(self._trie) and self._trie[j].startswith(prefix)


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
