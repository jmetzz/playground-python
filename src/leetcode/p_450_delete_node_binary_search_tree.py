"""https://leetcode.com/problems/delete-node-in-a-bst/description

450. Delete Node in a BST
Medium

Given a root node reference of a BST and a key,
delete the node with the given key in the BST.
Return the root node reference (possibly updated) of the BST.

Basically, the deletion can be divided into two stages:

Search for a node to remove.
If the node is found, delete the node.


Example 1:
Input: root = [5,3,6,2,4,null,7], key = 3
Output: [5,4,6,2,null,null,7]
Explanation: Given key to delete is 3. So we find the node
with value 3 and delete it.
One valid answer is [5,4,6,2,null,null,7].
Another valid answer is [5,2,6,null,4,null,7] and it's also accepted.

Example 2:
Input: root = [5,3,6,2,4,null,7], key = 0
Output: [5,3,6,2,4,null,7]
Explanation: The tree does not contain a node with value = 0.
Example 3:

Input: root = [], key = 0
Output: []


Constraints:

The number of nodes in the tree is in the range [0, 104].
-105 <= Node.val <= 105
Each node has a unique value.
root is a valid binary search tree.
-105 <= key <= 105

Follow up: Could you solve it with time complexity O(height of tree)?
"""

from typing import Optional

from data_structures.trees import BinaryTreeNode as TreeNode


def delete_node(root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
    if not root:
        return root
    # 1) search the target node
    if key > root.val:
        root.right = delete_node(root.right, key)
    elif key < root.val:
        root.left = delete_node(root.left, key)
    elif not root.left:
        return root.right
    elif not root.right:
        return root.left
    else:
        # target has 2 children case
        # find the min from the right subtree
        curr = root.right
        while curr.left:
            curr = curr.left
        root.val = curr.val  # replace the root value with the smallest in the
        # recusivelly delete the smallest value from the rigth subtree,
        # since we have just moved it to the current root
        root.right = delete_node(root.right, curr.val)
    return root


def highest_value_subtree(root: TreeNode):
    curr = root
    while curr.right:
        curr = curr.right
    return curr


def lowest_value_subtree(root: TreeNode):
    curr = root
    while curr.left:
        curr = curr.left
    return curr
