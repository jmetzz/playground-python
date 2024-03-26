"""
https://leetcode.com/problems/even-odd-tree/description

1609. Even Odd Tree
Medium

A binary tree is named Even-Odd if it meets the following conditions:

The root of the binary tree is at level index 0, its children are at level index 1,
their children are at level index 2, etc.

For every even-indexed level, all nodes at the level have odd integer values in
strictly increasing order (from left to right).

For every odd-indexed level, all nodes at the level have even integer values in
strictly decreasing order (from left to right).

Given the root of a binary tree, return true if the binary tree is Even-Odd, otherwise return false.



Example 1:
Input: root = [1,10,4,3,null,7,9,12,8,6,null,null,2]
Output: true
Explanation: The node values on each level are:
Level 0: [1]
Level 1: [10,4]
Level 2: [3,7,9]
Level 3: [12,8,6,2]
Since levels 0 and 2 are all odd and increasing and levels 1 and 3 are all even and decreasing, the tree is Even-Odd.

Example 2:
Input: root = [5,4,2,3,3,7]
Output: false
Explanation: The node values on each level are:
Level 0: [5]
Level 1: [4,2]
Level 2: [3,3,7]
Node values in level 2 must be in strictly increasing order, so the tree is not Even-Odd.

Example 3:
Input: root = [5,9,1,3,5,7]
Output: false
Explanation: Node values in the level 1 should be even integers.


Constraints:
The number of nodes in the tree is in the range [1, 105].
1 <= Node.val <= 106

"""

from collections import deque
from typing import Optional

from data_structures.trees import BinaryTreeNode


def is_even_odd_tree(root: Optional[BinaryTreeNode]) -> bool:
    if not root or root.val % 2 == 0:
        return False

    q = deque([root])
    is_even_level = True  # Starting with level 0, which should have odd values

    while q:
        # add all nodes in curr level to the queue
        level_length = len(q)
        prev_value = None

        for _ in range(level_length):
            # visit the node:
            # 1. check the constraints, and
            # 2. add their children to the queue
            node = q.popleft()
            # 1 >>>>
            if is_even_level:
                # even-indexed level: all nodes must have ODD values
                # in strictly INCREASING order (from left to right).
                if node.val % 2 == 0 or (prev_value is not None and prev_value >= node.val):
                    return False
            else:
                # odd-indexed level: all nodes must have EVEN values
                # in strictly DECREASING order (from left to right).
                if node.val % 2 != 0 or (prev_value is not None and prev_value <= node.val):
                    return False
            prev_value = node.val
            # 2 >>>>
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)

        is_even_level = not is_even_level
    return True


def run_for_tree(arr, show=False):
    tree = BinaryTreeNode.deserialize(arr)
    print(">>>")
    print(is_even_odd_tree(tree))
    if show:
        print(BinaryTreeNode.serialize(tree))
    print("<<<")
    print()


if __name__ == "__main__":
    # run_for_tree([1, 10, 4, 3, None, 7, 9, 12, 8, None, None, 6, None, None, 2], True)  # expect true
    # run_for_tree([5, 4, 2, 3, 3, 7], True)  # expect false
    # run_for_tree([5, 9, 1, 3, 5, 7],True)  # expect false

    # run_for_tree(
    # [11, 18, 14, 3, 7, None, None, None, None, 18,
    # None, None, None, None, None, None, None, None, None, 6], True) # false
    run_for_tree([1, 10, 4, 3, None, 7, 9, 12, 8, 6, None, None, 2], True)

    run_for_tree([11, 18, 14, 3, 7, None, None, None, None, 18, None, 6], True)  # badly formed tree
