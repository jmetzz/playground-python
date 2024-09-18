"""
https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/description/

235. Lowest Common Ancestor of a Binary Search Tree

Medium

Given a binary search tree (BST), find the lowest common ancestor (LCA) node
of two given nodes in the BST.

According to the definition of LCA on Wikipedia:
“The lowest common ancestor is defined between two nodes p and q as the
lowest node in T that has both p and q as descendants
(where we allow a node to be a descendant of itself).”

Example 1:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.

Example 2:
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

Example 3:
Input: root = [2,1], p = 2, q = 1
Output: 2


Constraints:

The number of nodes in the tree is in the range [2, 10^5].
-10^9 <= Node.val <= 10^9
All Node.val are unique.
p != q
p and q will exist in the BST.
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def lowest_common_ancestor_binary_search_tree(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    curr = root
    while curr:
        if p.val > curr.val and q.val > curr.val:
            # If both p and q are greater than the current node (curr.val),
            # it means that both nodes must lie in the right subtree
            curr = curr.right
        elif p.val < curr.val and q.val < curr.val:
            # If both p and q are smaller than the current node,
            # it means that both nodes are in the left subtree
            curr = curr.left
        else:
            # If one node is on the left and the other is on the right
            # (or one of them is the current node), then the
            # current node is the LCA, because it's the lowest point
            # where the paths to p and q diverge.
            return curr
    return None


if __name__ == "__main__":
    tree = TreeNode.from_array([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5])
    p_node = TreeNode.find_node_by_value(tree, 2)
    q_node = TreeNode.find_node_by_value(tree, 8)
    print(lowest_common_ancestor_binary_search_tree(tree, p_node, q_node).val)  # 6

    q_node = TreeNode.find_node_by_value(tree, 4)
    print(lowest_common_ancestor_binary_search_tree(tree, p_node, q_node).val)  # 2

    tree = TreeNode.from_array([2, 1])
    p_node = TreeNode.find_node_by_value(tree, 2)
    q_node = TreeNode.find_node_by_value(tree, 1)
    print(lowest_common_ancestor_binary_search_tree(tree, p_node, q_node).val)  # 2
