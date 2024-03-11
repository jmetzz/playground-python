"""
https://leetcode.com/problems/diameter-of-binary-tree/description

543. Diameter of Binary Tree
Easy

Given the root of a binary tree, return the length of the diameter of the tree.

The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
This path may or may not pass through the root.

The length of a path between two nodes is represented by the number of edges between them.


Example 1:
Input: root = [1,2,3,4,5]
Output: 3
Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].

Example 2:
Input: root = [1,2]
Output: 1

Constraints:
The number of nodes in the tree is in the range [1, 104].
-100 <= Node.val <= 100
"""

from collections import deque
from typing import Optional

from data_structures.trees import BinaryTreeNode


def dfs_height(r):
    if r is None or (r.left is None and r.right is None):
        # reached a leaf
        return 0
    l_height = dfs_height(r.left) + 1
    r_height = dfs_height(r.right) + 1
    return max(l_height, r_height)


def dfs_height_memo(r, memo_cache: dict):
    if r in memo_cache:
        return memo_cache[r]
    if r is None or (r.left is None and r.right is None):
        # reached a leaf
        memo_cache[r] = 0
        return 0
    l_height = dfs_height_memo(r.left, memo_cache) + 1
    r_height = dfs_height_memo(r.right, memo_cache) + 1
    memo_cache[r] = max(l_height, r_height)
    return memo_cache[r]


def diameter_of_bin_tree_recursive(root: Optional[BinaryTreeNode], debug: bool = False) -> int:
    q = deque()
    q.append(root)
    max_path = 0
    while q:
        curr_r = q.popleft()
        if debug:
            print(f"root({curr_r.val}) --> max_path: {max_path}")

        if curr_r.left:
            left = dfs_height(curr_r.left) + 1
            q.insert(0, curr_r.left)
        else:
            left = 0
        if curr_r.right:
            right = dfs_height(curr_r.right) + 1
            q.append(curr_r.right)
        else:
            right = 0

        max_path = max(max_path, left + right)
    return max_path


def diameter_of_bin_tree_memoization(root: Optional[BinaryTreeNode], debug: bool = False) -> int:
    memo = {}
    q = deque()
    q.append(root)
    max_path = 0
    while q:
        curr_r = q.popleft()
        if debug:
            print(f"root({curr_r.val}) --> max_path: {max_path}")

        if curr_r.left:
            left = dfs_height_memo(curr_r.left, memo) + 1
            q.insert(0, curr_r.left)
        else:
            left = 0
        if curr_r.right:
            right = dfs_height_memo(curr_r.right, memo) + 1
            q.append(curr_r.right)
        else:
            right = 0

        max_path = max(max_path, left + right)
    return max_path


def diameter_of_bin_tree_bottom_up(root: Optional[BinaryTreeNode]) -> int:
    # Initialize a list to hold the maximum diameter encountered
    # Make this an object to be able to modify inside the inner function
    _diameter = [0]

    def _dfs(node: BinaryTreeNode) -> int:
        """Calculates the height of the node and
        modify the diameter as a side effect
        """
        if not node:
            # assumes the height of null nodes is -1
            return -1
        # Recursively calculate the height of left and right subtrees
        h_left = _dfs(node.left)
        h_right = _dfs(node.right)

        # Update the maximum diameter encountered so far
        _diameter[0] = max(_diameter[0], h_left + h_right + 2)

        # Return the depth of the current node
        return 1 + max(h_left, h_right)

    # start from the root
    _ = _dfs(root)
    return _diameter[0]


if __name__ == "__main__":
    tree = BinaryTreeNode.deserialize([1, 2, 3, 4, 5])  # expect: 3
    # tree = TreeNode.build([1, 2, None, 3, 4, None, None, 5, None, None, 6])  # expect: 4
    # arr = [None] * 194
    # notes = [0, 1, 2, 5, 6, 11, 12, 13, 23, 25, 26, 47, 51, 53, 95, 96, 107, 192, 193]  # expected: 9
    # for i in notes:
    #     arr[i] = i
    # tree = TreeNode.build(arr)
    print(diameter_of_bin_tree_recursive(tree))
    print(diameter_of_bin_tree_memoization(tree))
    print(diameter_of_bin_tree_bottom_up(tree))
