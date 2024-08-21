"""https://leetcode.com/problems/sum-of-left-leaves/description/

404. Sum of Left Leaves
Easy

Given the root of a binary tree, return the sum of all left leaves.

A leaf is a node with no children. A left leaf is a leaf that is the left child of another node.



Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.

Example 2:
Input: root = [1]
Output: 0


Constraints:

The number of nodes in the tree is in the range [1, 1000].
-1000 <= Node.val <= 1000


"""

from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def sum_of_left_leaves(root: Optional[TreeNode]) -> int:
    if not root:
        return 0

    stack = [(root, False)]
    cum_sum = 0

    while stack:
        node, is_left = stack.pop()
        if is_left and not node.left and not node.right:
            cum_sum += node.val
            continue
        if node.right:
            stack.append((node.right, False))
        if node.left:
            stack.append((node.left, True))
    return cum_sum


# print(sum_of_left_leaves(TreeNode.deserialize([3])))
# print(sum_of_left_leaves(TreeNode.deserialize([3, 9, 20])))
# print(sum_of_left_leaves(TreeNode.deserialize([3, 9, 20, None, None, 15, 7])))
