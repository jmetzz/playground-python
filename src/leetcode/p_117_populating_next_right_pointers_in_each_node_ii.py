"""
https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description
117. Populating Next Right Pointers in Each Node II

Medium

Given a binary tree

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node.
If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.



Example 1:
Input: root = [1,2,3,4,5,null,7]
Output: [1,#,2,3,#,4,5,7,#]
Explanation: Given the above binary tree (Figure A), your function should
populate each next pointer to point to its next right node.
The serialized output is in level order as connected by the next pointers,
with '#' signifying the end of each level.

Example 2:
Input: root = []
Output: []

Constraints:
The number of nodes in the tree is in the range [0, 6000].
-100 <= Node.val <= 100


Follow-up:

You may only use constant extra space.
The recursive approach is fine. You may assume implicit stack space
does not count as extra space for this problem.
"""

from collections import deque
from typing import Self


class Node:
    def __init__(self, val: int = 0, left: Self = None, right: Self = None, nxt: Self = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = nxt


def connect(root: Node) -> Node:
    if root is None:
        return None

    queue = deque([root])
    while queue:
        n = len(queue)
        for i in range(n):
            curr = queue.popleft()
            if i < n - 1:
                # only link if it is not the right most node in the level
                curr.next = queue[0]
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)

    return root


def build_tree(arr: list[int], root: int = 0):
    if root >= len(arr) or arr[root] is None:
        return None

    return Node(arr[root], build_tree(arr, root * 2 + 1), build_tree(arr, root * 2 + 2))


def serialize_tree(root: Node) -> str:
    if not root:
        return ""

    result = []
    curr = root

    while curr:
        level_curr = curr
        next_level_start = None
        while level_curr:
            result.append(str(level_curr.val))
            # Find the first node in the next level if not already found
            if not next_level_start:
                next_level_start = level_curr.left or level_curr.right
            level_curr = level_curr.next
        result.append("#")

        # Move to the next level's start
        curr = next_level_start

    return ",".join(result)


def print_levels(root: Node):
    while root:
        curr = root
        while curr:
            print(curr.val, end=" -> " if curr.next else " -> None\n")
            curr = curr.next
        # Move to the next level
        if root.left:
            root = root.left
        elif root.right:
            root = root.right
        else:
            # Find the next level's starting node via the next pointers
            while root and not root.left and not root.right:
                root = root.next


if __name__ == "__main__":
    tree = connect(build_tree([1, 2, 3, 4, 5, None, 7]))
    print(serialize_tree(tree))
    print_levels(tree)

    print()
    tree = connect(build_tree([0]))
    print(serialize_tree(tree))
    print_levels(tree)
