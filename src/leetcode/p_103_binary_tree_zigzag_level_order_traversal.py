"""
https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/submissions/

103. Binary Tree Zigzag Level Order Traversal

Medium

Given the root of a binary tree, return the zigzag level order traversal of its nodes' values.
(i.e., from left to right, then right to left for the next level and alternate between).



Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[20,9],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 2000].
-100 <= Node.val <= 100
"""

from collections import deque

from data_structures.trees import BinaryTreeNode as TreeNode


def zigzag_level_order(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []

    left_to_right = True
    queue = deque([root])
    answer = []
    while queue:
        l_size, level_values = len(queue), []
        for _ in range(l_size):
            curr = queue.popleft()
            level_values.append(curr.val)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        answer.append(level_values if left_to_right else level_values[::-1])
        left_to_right = not left_to_right
    return answer


if __name__ == "__main__":
    tree = TreeNode.from_array([3, 9, 20, None, None, 15, 7])
    print(zigzag_level_order(tree))  # [[3], [20, 9], [15, 7]]

    tree = TreeNode.from_array([1])
    print(zigzag_level_order(tree))  # [[1]]

    tree = TreeNode.from_array([])
    print(zigzag_level_order(tree))  # []

    tree = TreeNode.from_array([1, 2, 3, 4, None, None, 5])
    print(zigzag_level_order(tree))  # [[1],[3,2],[4,5]]
