"""
https://leetcode.com/problems/kth-smallest-element-in-a-bst/description

230. Kth Smallest Element in a BST
Medium

Given the root of a binary search tree, and an integer k, return the kth smallest
value (1-indexed) of all the values of the nodes in the tree.


Example 1:
Input: root = [3,1,4,null,2], k = 1
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
Output: 3


Constraints:

The number of nodes in the tree is n.
1 <= k <= n <= 10^4
0 <= Node.val <= 10^4


Follow up: If the BST is modified often (i.e., we can do insert and delete operations)
and you need to find the kth smallest frequently, how would you optimize?

Hint 1: Try to utilize the property of a BST.
Hint 2: Try in-order traversal. (Credits to @chan13)
Hint 3: What if you could modify the BST node's structure?
Hint 4: The optimal runtime complexity is O(height of BST).
"""

from data_structures.trees import BinaryTreeNode as TreeNode


def kth_smallest_recur_inoder(root: TreeNode | None, k: int) -> int:
    order = []

    def _inorder(node):
        if not node:
            return
        _inorder(node.left)
        order.append(node.val)
        _inorder(node.right)

    _inorder(root)
    return order[k - 1]


def kth_smallest_recur_inoder_2(root: TreeNode | None, k: int) -> int:
    order = []

    def _inorder(node):
        if not node or len(order) >= k:
            return
        _inorder(node.left)
        order.append(node.val)
        _inorder(node.right)

    _inorder(root)
    return order[k - 1]


def kth_smallest_iter_stack_double_loop(root: TreeNode | None, k: int) -> int:
    if not root or k < 0:
        return -1

    curr = root
    stack = []
    while curr:
        # traverse to the smallest value
        stack.append(curr)
        curr = curr.left
    i = 0
    while stack and i < k:
        i += 1
        curr = stack.pop()
        nxt_smaller = curr.right
        while nxt_smaller:
            # traverse to the smallest value this this root
            stack.append(nxt_smaller)
            nxt_smaller = nxt_smaller.left

    answer = nxt_smaller or curr or root
    return answer.val


def kth_smallest_iter_stack_single_loop(root: TreeNode | None, k: int) -> int:
    stack = []
    curr = root

    while stack or curr:
        if curr:
            # traverse to the smallest node
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            if k == 1:
                break
            # explore the right sub-tree
            curr = curr.right
            k -= 1  # account the for curr element as one of the smaller elements

    return curr.val


if __name__ == "__main__":
    tree = TreeNode.from_array([3, 1, 4, None, 2])
    print(kth_smallest_iter_stack_double_loop(tree, 1))  # 1

    tree = TreeNode.from_array([5, 3, 6, 2, 4, None, None, 1])
    print(kth_smallest_iter_stack_double_loop(tree, 3))  # 3

    print("-------")
    tree = TreeNode.from_array([3, 1, 4, None, 2])
    print(kth_smallest_recur_inoder(tree, 1))  # 1

    tree = TreeNode.from_array([5, 3, 6, 2, 4, None, None, 1])
    print(kth_smallest_recur_inoder(tree, 3))  # 3

    print("-------")
    tree = TreeNode.from_array([3, 1, 4, None, 2])
    print(kth_smallest_recur_inoder_2(tree, 1))  # 1

    tree = TreeNode.from_array([5, 3, 6, 2, 4, None, None, 1])
    print(kth_smallest_recur_inoder_2(tree, 3))  # 3
