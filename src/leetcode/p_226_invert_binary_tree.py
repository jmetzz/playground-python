"""https://leetcode.com/problems/invert-binary-tree/description/

226. Invert Binary Tree
Easy
Given the root of a binary tree, invert the tree, and return its root.


Example 1:
Input: root = [4,2,7,1,3,6,9]
Output: [4,7,2,9,6,3,1]

Example 2:
Input: root = [2,1,3]
Output: [2,3,1]

Example 3:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
"""

from collections import deque

from data_structures.trees import BinaryTreeNode


def invert_tree(root: BinaryTreeNode | None) -> BinaryTreeNode | None:
    q = deque()
    q.append(root)
    while q:
        node = q.popleft()
        if not node:
            continue
        temp = node.left
        node.left, node.right = node.right, temp
        q.append(node.left)
        q.append(node.right)
    return root
