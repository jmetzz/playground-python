"""
https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description

106. Construct Binary Tree from Inorder and Postorder Traversal

Medium

Given two integer arrays inorder and postorder where inorder is the inorder
traversal of a binary tree and postorder is the postorder traversal of the same tree,
construct and return the binary tree.


Example 1:
Input: inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: inorder = [-1], postorder = [-1]
Output: [-1]


Constraints:
1 <= inorder.length <= 3000
postorder.length == inorder.length
-3000 <= inorder[i], postorder[i] <= 3000
inorder and postorder consist of unique values.
Each value of postorder also appears in inorder.
inorder is guaranteed to be the inorder traversal of the tree.
postorder is guaranteed to be the postorder traversal of the tree.

"""

from data_structures.trees import BinaryTreeNode as TreeNode


def build_tree(inorder: list[int], postorder: list[int]) -> TreeNode | None:
    if not inorder or not postorder:
        return None

    node = TreeNode(postorder[-1])

    mid = inorder.index(node.val)
    size_right_tree = len(inorder) - mid - 1

    post_end = len(postorder) - 1
    node.left = build_tree(inorder[:mid], postorder[:mid])
    node.right = build_tree(inorder[mid + 1 :], postorder[post_end - size_right_tree : post_end])

    return node


def build_tree_with_helper(inorder: list[int], postorder: list[int]) -> TreeNode | None:
    inorder_indices = {v: i for i, v in enumerate(inorder)}

    def _helper(in_start: int, in_end: int, post_start: int, post_end: int) -> TreeNode | None:
        if in_start > in_end or post_start > post_end:
            return None

        # The last element in postorder is the root node.
        node = TreeNode(postorder[post_end])

        # Find the index of the root in inorder to split left and right subtrees.
        mid = inorder_indices[node.val]
        size_left_tree = mid - in_start

        # inorder boundaries: in_start, mid - 1
        # postorder boundaries: post_start, post_start + size_left_tree - 1
        node.left = _helper(in_start, mid - 1, post_start, post_start + size_left_tree - 1)

        # inorder boundaries: mid + 1, in_end
        # postorder boundaries: post_start + size_left_tree, post_end - 1
        node.right = _helper(mid + 1, in_end, post_start + size_left_tree, post_end - 1)

        return node

    return _helper(0, len(inorder) - 1, 0, len(postorder) - 1)


if __name__ == "__main__":
    funcs = [build_tree, build_tree_with_helper]
    for func in funcs:
        t1 = func(inorder=[9, 3, 15, 20, 7], postorder=[9, 15, 7, 20, 3])
        print(t1.serialize())  # [3,9,20,None,None,15,7]

        t2 = func(inorder=[-1], postorder=[-1])
        print(t2.serialize())  # [-1]
        print()
