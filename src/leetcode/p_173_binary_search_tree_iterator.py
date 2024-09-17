"""
https://leetcode.com/problems/binary-search-tree-iterator/description

173. Binary Search Tree Iterator

Medium

Implement the BSTIterator class that represents an iterator over the
in-order traversal of a binary search tree (BST):

BSTIterator(TreeNode root) Initializes an object of the BSTIterator class.
The root of the BST is given as part of the constructor.
The pointer should be initialized to a non-existent number smaller
than any element in the BST.
boolean hasNext() Returns true if there exists a number in the traversal
to the right of the pointer, otherwise returns false.
int next() Moves the pointer to the right, then returns the number at the pointer.
Notice that by initializing the pointer to a non-existent smallest number,
the first call to next() will return the smallest element in the BST.

You may assume that next() calls will always be valid. That is,
there will be at least a next number in the in-order traversal when next() is called.



Example 1:
Input
["BSTIterator", "next", "next", "hasNext", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
[[[7, 3, 15, null, null, 9, 20]], [], [], [], [], [], [], [], [], []]
Output
[null, 3, 7, true, 9, true, 15, true, 20, false]

Explanation
BSTIterator bSTIterator = new BSTIterator([7, 3, 15, null, null, 9, 20]);
bSTIterator.next();    // return 3
bSTIterator.next();    // return 7
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 9
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 15
bSTIterator.hasNext(); // return True
bSTIterator.next();    // return 20
bSTIterator.hasNext(); // return False


Constraints:

The number of nodes in the tree is in the range [1, 10^5].
0 <= Node.val <= 10^6
At most 105 calls will be made to hasNext, and next.


Follow up:

Could you implement next() and hasNext() to run in average
O(1) time and use O(h) memory, where h is the height of the tree?
"""

from data_structures.trees import BinaryTreeNode as TreeNode


class BSTIterator:
    def __init__(self, root: TreeNode | None):
        self._stack = []
        self._stack_branch(root)

    def _stack_branch(self, node: TreeNode) -> None:
        while node:
            self._stack.append(node)
            node = node.left

    def next(self) -> int:
        nxt = self._stack.pop()
        self._stack_branch(nxt.right)
        return nxt.val

    def hasNext(self) -> bool:
        return self._stack != []


class BSTIterator2:
    def __init__(self, root: TreeNode | None):
        self.nxt_index = 0
        self.values = []
        self._in_order(root)

    def next(self) -> int:
        self.nxt_index += 1
        return self.values[self.nxt_index - 1]

    def hasNext(self) -> bool:
        return self.nxt_index < len(self.values)

    def _in_order(self, root: TreeNode | None) -> None:
        if not root:
            return
        self._in_order(root.left)
        self.values.append(root.val)
        self._in_order(root.right)


if __name__ == "__main__":
    tree = TreeNode.deserialize([7, 3, 15, None, None, 9, 20])

    bst_iterator = BSTIterator(tree)
    print(bst_iterator.next())  # return 3
    print(bst_iterator.next())  # return 7
    print(bst_iterator.hasNext())  # return True
    print(bst_iterator.next())  # return 9
    print(bst_iterator.hasNext())  # return True
    print(bst_iterator.next())  # return 15
    print(bst_iterator.hasNext())  # return True
    print(bst_iterator.next())  # return 20
    print(bst_iterator.hasNext())  # return False

    bst_iterator = BSTIterator(None)
    print(bst_iterator.hasNext())  # return False
