"""
https://leetcode.com/problems/maximum-level-sum-of-a-binary-tree/description

1161. Maximum Level Sum of a Binary Tree
Medium

Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.

Return the smallest level x such that the sum of all the values of nodes at level x is maximal.

Example 1:
Input: root = [1,7,0,7,-8,null,null]
Output: 2
Explanation:
Level 1 sum = 1.
Level 2 sum = 7 + 0 = 7.
Level 3 sum = 7 + -8 = -1.
So we return the level with the maximum sum which is level 2.

Example 2:
Input: root = [989,null,10250,98693,-89388,null,null,null,-32127]
Output: 2

Constraints:
The number of nodes in the tree is in the range [1, 104].
-105 <= Node.val <= 105
"""

from collections import deque
from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def max_level_sum_1(root: Optional[TreeNode]) -> int:
    """
    Finds the level of a binary tree with the maximum sum of values.

    Traverses the tree in level order (breadth-first search) to
    compute the sum of values at each level,
    then returns the level that has the maximum sum.
    If the tree is empty, returns 0.

    Args:
        root (Optional[TreeNode]): The root node of the binary tree.

    Returns:
        int: The level (1-indexed) of the binary tree with the
        maximum sum of values. If the tree is empty, returns 0.

    Examples:
        Given a binary tree:

            1
           / \
          7   0
         / \
        7  -8

        The function would return 2 since the second level has the
        maximum sum of values (7 + 0 = 7).

        Given a binary tree:

            7
           / \
          7  -8
            /
          -7

        The function would return 3, where the third level
        (with value -7) has the maximum sum of values.
    """

    if not root:
        return 0

    nodes_to_explore = deque([root])
    max_sofar = float("-inf")
    level = 0
    max_level = 0
    while nodes_to_explore:
        level_sum = 0
        level += 1
        for _ in range(len(nodes_to_explore)):
            node = nodes_to_explore.popleft()
            level_sum += node.val
            if node.left:
                nodes_to_explore.append(node.left)
            if node.right:
                nodes_to_explore.append(node.right)
        if level_sum > max_sofar:
            max_sofar = level_sum
            max_level = level
    return max_level


def max_level_sum_2(root: Optional[TreeNode]) -> int:
    """
    Finds the level of a binary tree with the maximum sum of its node values.

    This function traverses the tree in a breadth-first manner,
    calculating the sum of values at each level and then returns the
    level (1-indexed) with the maximum sum. If the tree is empty, it returns 0.

    Args:
        root (Optional[TreeNode]): The root node of the binary tree.

    Returns:
        int: The level of the tree with the maximum sum. If the tree is empty, returns 0.

    Example:
        Given a binary tree:
            1
           / \
          2   3
         / \   \
        4   5   8
               / \
              6   7
        max_level_sum(root) -> 4 (because the sum at level 4 is 6+7=13, which is the maximum).
    """

    if not root:
        return 0

    q = deque([(root, 1)])  # Store nodes along with their level
    level_sums = {}

    while q:
        node, level = q.popleft()
        level_sums[level] = level_sums.get(level, 0) + node.val

        for child in [node.left, node.right]:
            if child:
                q.append((child, level + 1))

    # Find the level with the maximum sum
    max_level = max(level_sums, key=level_sums.get)
    return max_level
