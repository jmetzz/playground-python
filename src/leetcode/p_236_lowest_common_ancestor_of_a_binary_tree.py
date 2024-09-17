"""

236. Lowest Common Ancestor of a Binary Tree

Medium
Topics
Companies
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia:
“The lowest common ancestor is defined between two nodes p and q as the
lowest node in T that has both p and q as descendants
(where we allow a node to be a descendant of itself).”



Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a
descendant of itself according to the LCA definition.

Example 3:
Input: root = [1,2], p = 1, q = 2
Output: 1


Constraints:

The number of nodes in the tree is in the range [2, 105].
-10^9 <= Node.val <= 10^9
All Node.val are unique.
p != q
p and q will exist in the tree.
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def lowest_common_ancestor_generic_binary_tree(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    # Base case: if root is null, or root is p, or root is q
    if not root or root == p or root == q:
        return root

    # Recurse on the left and right subtree
    left = lowest_common_ancestor_generic_binary_tree(root.left, p, q)
    right = lowest_common_ancestor_generic_binary_tree(root.right, p, q)

    # If both the left and right subtrees return non-null values,
    # that means both p and q are found in different subtrees of this node,
    # and hence this node is the LCA
    if left and right:
        return root

    # Otherwise, return the non-null child (either from left or right)
    # If one of the subtrees returns a non-null value,
    # that means either p or q is found in that subtree,
    # so we return that value up to the parent.
    return left if left else right


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


def construct_tree_from_list(lst: list[int]) -> TreeNode:
    """Function to construct a binary tree from a list (level-order)"""
    if not lst:
        return None

    root = TreeNode(lst[0])
    queue = [root]
    i = 1

    while queue and i < len(lst):
        node = queue.pop(0)

        if i < len(lst) and lst[i] is not None:
            node.left = TreeNode(lst[i])
            queue.append(node.left)
        i += 1

        if i < len(lst) and lst[i] is not None:
            node.right = TreeNode(lst[i])
            queue.append(node.right)
        i += 1

    return root


def find_node_by_value(root: TreeNode, val: int):
    """Helper function to find the reference to a node by its value"""
    if not root:
        return None
    if root.val == val:
        return root

    left_search = find_node_by_value(root.left, val)
    if left_search:
        return left_search

    return find_node_by_value(root.right, val)


if __name__ == "__main__":
    tree = construct_tree_from_list([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4])

    p_node = find_node_by_value(tree, 5)
    q_node = find_node_by_value(tree, 1)
    print(lowest_common_ancestor_generic_binary_tree(tree, p_node, q_node).val)  # 3

    q_node = find_node_by_value(tree, 4)
    print(lowest_common_ancestor_generic_binary_tree(tree, p_node, q_node).val)  # 5

    tree = TreeNode.deserialize([1, 2])
    p_node = find_node_by_value(tree, 1)
    q_node = find_node_by_value(tree, 2)

    print(lowest_common_ancestor_generic_binary_tree(tree, p_node, q_node).val)  # 1
