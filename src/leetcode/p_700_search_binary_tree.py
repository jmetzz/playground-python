"""https://leetcode.com/problems/search-in-a-binary-search-tree/description
700. Search in a Binary Search Tree
Easy

You are given the root of a binary search tree (BST) and an integer val.

Find the node in the BST that the node's value equals val and return
the subtree rooted with that node. If such a node does not exist, return null.



Example 1:
Input: root = [4,2,7,1,3], val = 2
Output: [2,1,3]

Example 2:
Input: root = [4,2,7,1,3], val = 5
Output: []

Constraints:
The number of nodes in the tree is in the range [1, 5000].
1 <= Node.val <= 107
root is a binary search tree.
1 <= val <= 107
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def search_bst(root: TreeNode | None, val: int) -> TreeNode | None:
    node = root
    while node:
        if not node:
            return None
        if node.val == val:
            return node
        if val < node.val:
            node = node.left
        else:
            node = node.right
    return None


def search_bst_recursive(root: TreeNode | None, val: int) -> TreeNode | None:
    if root is None:
        return None
    if root.val == val:
        return root
    if root.val < val:
        return search_bst_recursive(root.right, val)
    return search_bst_recursive(root.left, val)
