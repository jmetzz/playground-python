"""
https://leetcode.com/problems/binary-tree-level-order-traversal/

102. Binary Tree Level Order Traversal

Medium

Given the root of a binary tree, return the level order traversal of
its nodes' values. (i.e., from left to right, level by level).


Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]

Example 2:
Input: root = [1]
Output: [[1]]

Example 3:
Input: root = []
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 2000].
-1000 <= Node.val <= 1000

"""

from collections import deque

from data_structures.trees import BinaryTreeNode as TreeNode


def level_order(root: TreeNode | None) -> list[list[int]]:
    if not root:
        return []

    queue = deque([root])
    answer = []

    while queue:
        level_size, level_values = len(queue), []
        for _ in range(level_size):
            curr = queue.popleft()
            level_values.append(curr.val)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        answer.append(level_values)

    return answer


if __name__ == "__main__":
    tree = TreeNode.from_array([3, 9, 20, None, None, 15, 7])
    print(level_order(tree))  # [[3],[9,20],[15,7]]

    print(level_order(TreeNode.from_array([1])))  # [[1]]

    print(level_order(TreeNode.from_array([])))  # []
