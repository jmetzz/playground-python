"""https://leetcode.com/problems/pseudo-palindromic-paths-in-a-binary-tree/description/

1457. Pseudo-Palindromic Paths in a Binary Tree
#Medium

Given a binary tree where node values are digits from 1 to 9.
A path in the binary tree is said to be pseudo-palindromic if
at least one permutation of the node values in the path is a palindrome.

Return the number of pseudo-palindromic paths going from the root node to leaf nodes.


Example 1:
Input: root = [2,3,1,3,1,null,1]
Output: 2
Explanation: The figure above represents the given binary tree.
There are three paths going from the root node to leaf nodes:
the red path [2,3,3], the green path [2,1,1], and the path [2,3,1].
Among these paths only red path and green path are pseudo-palindromic paths
since the red path [2,3,3] can be rearranged in [3,2,3] (palindrome) and
the green path [2,1,1] can be rearranged in [1,2,1] (palindrome).

Example 2:
Input: root = [2,1,1,1,3,null,null,null,null,null,1]
Output: 1
Explanation: The figure above represents the given binary tree.
There are three paths going from the root node to leaf nodes:
the green path [2,1,1], the path [2,1,3,1], and the path [2,1].
Among these paths only the green path is pseudo-palindromic
since [2,1,1] can be rearranged in [1,2,1] (palindrome).

Example 3:
Input: root = [9]
Output: 1

Constraints:
The number of nodes in the tree is in the range [1, 105].
1 <= Node.val <= 9
"""

from data_structures.trees import BinaryTreeNode


class PseudoPalindromicPaths_iterative:
    def solve(self, root: BinaryTreeNode) -> int:
        count = 0
        stack = [(root, 0)]
        while stack:
            node, path = stack.pop()
            if node:
                path ^= 1 << node.val  # freq of each digit
                if not node.left and not node.right:
                    # if a leaf is reached:
                    if path & (path - 1) == 0:
                        count += 1
                else:
                    stack.append((node.right, path))
                    stack.append((node.left, path))
        return count


class PseudoPalindromicPaths_recursive:
    def solve(self, root: BinaryTreeNode) -> int:
        return self._count_solutions(root, 0)

    def _count_solutions(self, node: BinaryTreeNode, path) -> int:
        if not node:
            return 0

        path ^= 1 << node.val
        if not node.left and not node.right:
            return 1 if path & (path - 1) == 0 else 0

        return self._count_solutions(node.left, path) + self._count_solutions(node.right, path)


def df_traverse(node: BinaryTreeNode) -> list:
    return [node.val] + df_traverse(node.left) + df_traverse(node.right) if node else []


if __name__ == "__main__":
    tree = BinaryTreeNode.deserialize([2, 3, 1, 3, 1, None, 1], 0)
    print(PseudoPalindromicPaths_iterative().solve(tree))
    print(PseudoPalindromicPaths_recursive().solve(tree))
