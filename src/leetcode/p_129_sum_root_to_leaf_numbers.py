"""https://leetcode.com/problems/sum-root-to-leaf-numbers/description

129. Sum Root to Leaf Numbers
Medium

You are given the root of a binary tree containing digits from 0 to 9 only.

Each root-to-leaf path in the tree represents a number.

For example, the root-to-leaf path 1 -> 2 -> 3 represents the number 123.
Return the total sum of all root-to-leaf numbers. Test cases are generated
so that the answer will fit in a 32-bit integer.

A leaf node is a node with no children.



Example 1:


Input: root = [1,2,3]
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.
Example 2:


Input: root = [4,9,0,5,1]
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.


Constraints:

The number of nodes in the tree is in the range [1, 1000].
0 <= Node.val <= 9
The depth of the tree will not exceed 10.
"""

from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def sum_root_to_leaf_path(root: Optional[TreeNode]) -> int:
    def _dfs(node, accumulator):
        if not node:
            return 0

        accumulator = accumulator * 10 + node.val
        if not node.left and not node.right:
            return accumulator

        return _dfs(node.left, accumulator) + _dfs(node.right, accumulator)

    return _dfs(root, 0)


if __name__ == "__main__":
    print(sum_root_to_leaf_path(TreeNode.deserialize([1, 2, 3])))  # 25
    print(sum_root_to_leaf_path(TreeNode.deserialize([4, 9, 0, 5, 1])))  # 1026
