"""https://leetcode.com/problems/smallest-string-starting-from-leaf/description/

988. Smallest String Starting From Leaf
Medium

You are given the root of a binary tree where each node has a value in
the range [0, 25] representing the letters 'a' to 'z'.

Return the lexicographically smallest string that starts at a leaf of
this tree and ends at the root.

As a reminder, any shorter prefix of a string is lexicographically smaller.

For example, "ab" is lexicographically smaller than "aba".
A leaf of a node is a node that has no children.


Example 1:
Input: root = [0,1,2,3,4,3,4]
Output: "dba"

Example 2:
Input: root = [25,1,3,1,3,0,2]
Output: "adz"

Example 3:
Input: root = [2,2,1,null,1,0,null,0]
Output: "abc"


Constraints:

The number of nodes in the tree is in the range [1, 8500].
0 <= Node.val <= 25
"""

from collections import deque
from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def smallest_from_leaf_dfs(root: Optional[TreeNode]) -> str:
    def dfs_helper(node, curr_str):
        nonlocal shortest_str
        if not node:
            return

        curr_str = chr(node.val + 97) + curr_str  # ord('a') == 97
        if not node.left and not node.right:  # noqa: SIM102
            if not shortest_str or curr_str < shortest_str:
                shortest_str = curr_str
        if node.left:
            dfs_helper(node.left, curr_str)
        if node.right:
            dfs_helper(node.right, curr_str)

    shortest_str = ""
    dfs_helper(root, "")
    return shortest_str


def smallest_from_leaf_bfs(root: Optional[TreeNode]) -> str:
    smallest_string = ""
    node_queue = deque()
    a_offset = ord("a")

    # Add root node to deque along with its value converted to a character
    node_queue.append([root, chr(root.val + a_offset)])

    while node_queue:
        node, current_string = node_queue.popleft()

        # If current node is a leaf node
        if not node.left and not node.right:
            # Update smallest_string if it's empty or current string is smaller
            smallest_string = min(smallest_string, current_string) if smallest_string else current_string

        if node.left:
            node_queue.append([node.left, chr(node.left.val + a_offset) + current_string])

        if node.right:
            node_queue.append([node.right, chr(node.right.val + a_offset) + current_string])

    return smallest_string
