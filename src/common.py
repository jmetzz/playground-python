from typing import List, Optional

import numpy as np


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    @staticmethod
    def build(arr: List[Optional[int]], root: int = 0):
        if root >= len(arr) or arr[root] is None:
            return None

        return TreeNode(
            arr[root],
            TreeNode.build(arr, root * 2 + 1),
            TreeNode.build(arr, root * 2 + 2),
        )

    @staticmethod
    def from_indices(indices: List[int]):
        size = max(indices) + 1
        arr = np.full(size, None)
        arr[indices] = indices
        return TreeNode.build(arr.tolist())


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
