"""https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/description/

https://www.youtube.com/watch?v=ihj4IQGZ2zc

105. Construct Binary Tree from Preorder and Inorder Traversal

Medium

Given two integer arrays preorder and inorder where preorder is the preorder
traversal of a binary tree and inorder is the inorder traversal of the same tree,
 construct and return the binary tree.



Example 1:
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]

Example 2:
Input: preorder = [-1], inorder = [-1]
Output: [-1]


Constraints:
1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.

"""

import timeit

from data_structures.trees import BinaryTreeNode as TreeNode


def build_tree(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    if not preorder or not inorder:
        return None
    # preorder=[3, 9, 20, 15, 7]
    # inorder=[9, 3, 15, 20, 7]

    # The first element in preorder is the root node.
    node = TreeNode(preorder[0])  # node = [3]

    # Find the index of the root in inorder to split left and right subtrees.
    mid = inorder.index(preorder[0])  # 1

    # Recursively build the left and right subtrees.
    node.left = build_tree(preorder[1 : mid + 1], inorder[:mid])  # preorder=[9, 20), inorder=[9, 3)
    node.right = build_tree(preorder[mid + 1 :], inorder[mid + 1 :])  # preorder=[20, 15, 7], inorder=[15, 20, 7]
    return node


def build_tree_with_helper(preorder: list[int], inorder: list[int]) -> TreeNode | None:
    inorder_indices = {v: i for i, v in enumerate(inorder)}

    def _helper(pre_start: int, pre_end: int, in_start: int, in_end: int) -> TreeNode | None:
        if pre_start > pre_end or in_start > in_end:
            return None

        # The first element in preorder is the root node.
        root_val = preorder[pre_start]
        root = TreeNode(root_val)

        # Find the index of the root in inorder to split left and right subtrees.
        mid = inorder_indices[root_val]

        # Calculate the number of elements in the left subtree.
        left_tree_size = mid - in_start

        # Recursively build the left and right subtrees.
        root.left = _helper(pre_start + 1, pre_start + left_tree_size, in_start, mid - 1)
        root.right = _helper(pre_start + left_tree_size + 1, pre_end, mid + 1, in_end)

        return root

    # Start the recursion with the full range of preorder and inorder arrays.
    return _helper(0, len(preorder) - 1, 0, len(inorder) - 1)


if __name__ == "__main__":
    t1 = build_tree(preorder=[3, 9, 20, 15, 7], inorder=[9, 3, 15, 20, 7])
    print(t1.serialize())  # [3,9,20,None,None,15,7]

    t2 = build_tree_with_helper(preorder=[3, 9, 20, 15, 7], inorder=[9, 3, 15, 20, 7])
    print(t2.serialize())  # [3,9,20,None,None,15,7]

    preorder = [3, 9, 20, 15, 7]
    inorder = [9, 3, 15, 20, 7]
    # Assuming your original function is named build_tree_original
    time_original = timeit.timeit(
        stmt="build_tree(preorder, inorder)",
        setup="from __main__ import build_tree, preorder, inorder",
        number=10000,  # Adjust the number of iterations as needed
    )
    print(f"Original function time: {time_original:.5f} seconds")

    time_helper = timeit.timeit(
        stmt="build_tree_with_helper(preorder, inorder)",
        setup="from __main__ import build_tree_with_helper, preorder, inorder",
        number=10000,  # Adjust the number of iterations as needed
    )
    print(f"Helper function time: {time_helper:.5f} seconds")
