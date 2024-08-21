"""https://leetcode.com/problems/binary-tree-right-side-view/description

199. Binary Tree Right Side View
Medium

Given the root of a binary tree, imagine yourself standing on the right side of it,
return the values of the nodes you can see ordered from top to bottom.

Example 1:
Input: root = [1,2,3,null,5,null,4]
Output: [1,3,4]

Example 2:
Input: root = [1,null,3]
Output: [1,3]

Example 3:
Input: root = []
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
"""

from ast import List
from collections import deque
from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def right_side_view(root: Optional[TreeNode]) -> List[int]:
    if not root:
        return []
    right_side = [root.val]
    q = deque([root])
    while q:
        level_nodes = []
        for _ in range(len(q)):
            node = q.popleft()
            for child in (node.left, node.right):
                if child:
                    level_nodes.append(child)
        if level_nodes:
            right_side.append(level_nodes[-1].val)
            q.extend(level_nodes)
    return right_side
