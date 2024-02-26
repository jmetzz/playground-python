"""
https://leetcode.com/problems/same-tree/description/

100. Same Tree
Easy

Given the roots of two binary trees p and q, write a function to check
if they are the same or not.

Two binary trees are considered the same if they are structurally identical,
and the nodes have the same value.

Example 1:
Input: p = [1,2,3], q = [1,2,3]
Output: true

Example 2:
Input: p = [1,2], q = [1,null,2]
Output: false

Example 3:
Input: p = [1,2,1], q = [1,1,2]
Output: false


Constraints:
The number of nodes in both trees is in the range [0, 100].
-104 <= Node.val <= 104

"""
from typing import Optional

from common import TreeNode


def is_same_tree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if p is None and q is None:
        return True

    if p is not None and q is not None and p.val == q.val:
        return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
    else:
        return False


if __name__ == "__main__":
    print(is_same_tree(TreeNode.build([1, 2, 3]), TreeNode.build([1, 2, 3])))
