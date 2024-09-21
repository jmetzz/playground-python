"""
https://leetcode.com/problems/validate-binary-search-tree

98. Validate Binary Search Tree
Medium

Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:

* The left subtree of a node contains only nodes with keys less than the node's key.
* The right subtree of a node contains only nodes with keys greater than the node's key.
* Both the left and right subtrees must also be binary search trees.


Example 1:
Input: root = [2,1,3]
Output: true

Example 2:
Input: root = [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.


Constraints:
The number of nodes in the tree is in the range [1, 10^4].
-2^31 <= Node.val <= 2^31 - 1
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def is_valid_bst(root: TreeNode | None) -> bool:
    def _smallest_value(node) -> int:
        curr = node
        while curr.left:
            curr = curr.left
        return curr.val

    def _biggest_value(node) -> int:
        curr = node
        while curr.right:
            curr = curr.right
        return curr.val

    def _dfs_validate(node) -> bool:
        if not node.left and not node.right:
            # leaf nodes are valid BSTs
            return True

        if node.left:
            valid_left_subtree = _biggest_value(node.left) < node.val and _dfs_validate(node.left)
        else:
            valid_left_subtree = True

        if node.right:
            valid_right_subtree = _smallest_value(node.right) > node.val and _dfs_validate(node.right)
        else:
            valid_right_subtree = True

        return valid_left_subtree and valid_right_subtree

    if not root:
        return False

    return _dfs_validate(root)


def is_valid_bst_2(root: TreeNode | None) -> bool:
    def _dfs_validate(node, min_val, max_val):
        if not node:
            return True

        if not (min_val < node.val < max_val):
            return False

        valid_left_subtree = _dfs_validate(node.left, min_val, node.val)
        valid_right_subtree = _dfs_validate(node.right, node.val, max_val)

        return valid_left_subtree and valid_right_subtree

    if not root:
        return False

    return _dfs_validate(root, min_val=-float("inf"), max_val=float("inf"))


if __name__ == "__main__":
    tree = TreeNode.from_array([2, 1, 3])
    print(is_valid_bst(tree))  # True

    tree = TreeNode.from_array([5, 1, 4, None, None, 3, 6])
    print(is_valid_bst(tree))  # False

    tree = TreeNode.from_array([5, 4, 6, None, None, 3, 7])
    print(is_valid_bst(tree))  # False
