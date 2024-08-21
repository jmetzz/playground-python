"""https://leetcode.com/problems/symmetric-tree/description/
101. Symmetric Tree
Easy

Given the root of a binary tree, check whether it is a mirror of itself
(i.e., symmetric around its center).


Example 1:
Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:
Input: root = [1,2,2,null,3,null,3]
Output: false


Constraints:

The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100

Follow up: Could you solve it both recursively and iteratively?
"""

from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def is_symmetric_recursive(root: Optional[TreeNode]) -> bool:
    def _helper(root1, root2) -> bool:
        if root1 is None and root2 is None:
            return True
        if (root1 is None) != (root2 is None) or root1.val != root2.val:
            return False
        return _helper(root1.left, root2.right) and _helper(root1.right, root2.left)

    if root is None:
        return True
    return _helper(root.left, root.right)


def is_symmetric_iterative(root: Optional[TreeNode]) -> bool:
    """Note: missing nodes are not considered"""
    if root is None:
        return True
    q = [root.left, root.right]
    while q:
        l_tree = q.pop(0)
        r_tree = q.pop(0)

        if l_tree is None and r_tree is None:
            continue
        if not l_tree or not r_tree or l_tree.val != r_tree.val:
            return False
        q.extend([l_tree.left, r_tree.right, l_tree.right, r_tree.left])

    return True


if __name__ == "__main__":
    print(is_symmetric_recursive(TreeNode.deserialize([1, 2, 3])))
    print(is_symmetric_recursive(TreeNode.deserialize([1, 2, 2, 3, 4, 4, 3])))
    print(is_symmetric_recursive(TreeNode.deserialize([1, 2, 2, None, 3, None, 3])))
    print(is_symmetric_recursive(TreeNode.deserialize([1, 2, 2, 3, None, None, 3, 4, None, None, 4])))

    print("-----")
    print(is_symmetric_iterative(TreeNode.deserialize([1, 2, 3])))
    print(is_symmetric_iterative(TreeNode.deserialize([1, 2, 2, 3, 4, 4, 3])))
    print(is_symmetric_iterative(TreeNode.deserialize([1, 2, 2, None, 3, None, 3])))
    print(is_symmetric_iterative(TreeNode.deserialize([1, 2, 2, 3, None, None, 3, 4, None, None, 4])))
