"""https://leetcode.com/problems/maximum-depth-of-binary-tree/description

104. Maximum Depth of Binary Tree
Easy

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along
the longest path from the root node down to the farthest leaf node.



Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2


Constraints:

The number of nodes in the tree is in the range [0, 104].
-100 <= Node.val <= 100
"""

from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def max_depth_forward_accumulation(root: Optional[TreeNode]) -> int:
    def _dfs(node, curr_depth):
        if not node:
            return 0
        if not node.left and not node.right:
            # current node is a leaf
            return curr_depth + 1
        return max(_dfs(node.left, curr_depth + 1), _dfs(node.right, curr_depth + 1))

    return _dfs(root, 0)


def max_depth_backtrack_accumulatin(root: Optional[TreeNode]) -> int:
    def dfs_backtrack(node):
        if not node:
            return 0
        return 1 + max(dfs_backtrack(node.left), dfs_backtrack(node.right))

    return dfs_backtrack(root)


if __name__ == "__main__":
    tree = TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    print(max_depth_forward_accumulation(tree))
    print(max_depth_forward_accumulation(TreeNode.deserialize([1, None, 2])))

    print("---")
    print(max_depth_backtrack_accumulatin(tree))
    print(max_depth_backtrack_accumulatin(TreeNode.deserialize([1, None, 2])))
