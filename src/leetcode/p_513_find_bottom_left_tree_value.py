"""https://leetcode.com/problems/find-bottom-left-tree-value/description

513. Find Bottom Left Tree Value
Medium

Given the root of a binary tree, return the leftmost value in the last row of the tree.


Example 1:
Input: root = [2,1,3]
Output: 1

Example 2:
Input: root = [1,2,3,4,null,5,6,null,null,7]
Output: 7

Constraints:
The number of nodes in the tree is in the range [1, 104].
-231 <= Node.val <= 231 - 1
"""

from collections import deque
from math import inf
from typing import Optional

from conftest import TREE_0, TREE_1, TREE_2, TREE_3, TREE_4, TREE_5, TREE_6

from data_structures.trees import BinaryTreeNode


def find_bottom_left_value_iterative(root: Optional[BinaryTreeNode]) -> int:
    q = deque()
    q.append(root)
    answer = inf
    while q:
        collected = False
        for _ in range(len(q)):
            node = q.popleft()
            if node:
                q.append(node.left)
                q.append(node.right)
                if not collected:
                    collected = True
                    answer = node.val

    return answer


def find_bottom_left_value_recursive(root: Optional[BinaryTreeNode]) -> Optional[int]:
    depth = [-1]
    answer = [None]

    def _bfs(node, curr_depth):
        if not node:
            return
        if curr_depth > depth[0]:
            depth[0] = curr_depth
            answer[0] = node.val
        _bfs(node.left, curr_depth + 1)
        _bfs(node.right, curr_depth + 1)

    _bfs(root, 0)
    return answer[0]


if __name__ == "__main__":
    trees = [TREE_0, TREE_1, TREE_2, TREE_3, TREE_4, TREE_5, TREE_6]
    print([find_bottom_left_value_iterative(t) for t in trees])
    print([find_bottom_left_value_recursive(t) for t in trees])
