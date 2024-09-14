"""
https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description

114. Flatten Binary Tree to Linked List

Medium

Given the root of a binary tree, flatten the tree into a "linked list":

The "linked list" should use the same TreeNode class where the right child
pointer points to the next node in the list and the left child pointer is
always null.
The "linked list" should be in the same order as a pre-order traversal
of the binary tree.


Example 1:
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]

Example 2:
Input: root = []
Output: []

Example 3:
Input: root = [0]
Output: [0]


Constraints:

The number of nodes in the tree is in the range [0, 2000].
-100 <= Node.val <= 100


Hint 1:
If you notice carefully in the flattened tree, each node's right child points
to the next node of a pre-order traversal.

Follow up: Can you flatten the tree in-place (with O(1) extra space)?
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def flatten(root: TreeNode | None) -> None:
    """
    Use pre-order traversal.

    Do not return anything, modify root in-place instead.
    """

    def _dfs_flatter(curr) -> TreeNode:
        if not curr:
            return None

        left_tail = _dfs_flatter(curr.left)
        right_tail = _dfs_flatter(curr.right)

        if curr.left:
            left_tail.right = curr.right
            curr.right = curr.left
            curr.left = None

        # returns the tail node (the right most node)
        return right_tail or left_tail or curr

    _dfs_flatter(root)


if __name__ == "__main__":
    flatten(None)  # a list [] doesn't break the execution

    tree = TreeNode.deserialize([1])
    flatten(tree)
    print(tree.serialize_tree())
    print(tree.serialize())

    tree = TreeNode.deserialize([1, 2, 5, 3, 4, None, 6])
    flatten(tree)
    print(tree.serialize_tree())
    print(tree.serialize())
