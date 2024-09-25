"""https://leetcode.com/problems/add-one-row-to-tree/description

Given the root of a binary tree and two integers val and depth,
add a row of nodes with value val at the given depth depth.

Note that the root node is at depth 1.

The adding rule is:

Given the integer depth, for each not null tree node cur at the depth depth - 1,
create two tree nodes with value val as cur's left subtree root and right subtree root.
cur's original left subtree should be the left subtree of the new left subtree root.
cur's original right subtree should be the right subtree of the new right subtree root.
If depth == 1 that means there is no depth depth - 1 at all, then create a tree node
with value val as the new root of the whole original tree, and the original
tree is the new root's left subtree.

Example 1:
Input: root = [4,2,6,3,1,5], val = 1, depth = 2
Output: [4,1,1,2,null,null,6,3,1,5]

Example 2:
Input: root = [4,2,null,3,1], val = 1, depth = 3
Output: [4,2,null,1,1,3,null,null,1]


Constraints:

The number of nodes in the tree is in the range [1, 10**4].
The depth of the tree is in the range [1, 10**4].
-100 <= Node.val <= 100
-10**5 <= val <= 10**5
1 <= depth <= the depth of tree + 1

"""

from collections import deque

from data_structures.trees import BinaryTreeNode as TreeNode


def add_one_row(root: TreeNode | None, val: int, depth: int) -> TreeNode | None:
    if not root:
        return None
    if depth == 1:
        return TreeNode(val, left=root)

    queue = deque([root])
    level = 1
    while level < depth - 1 and queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        level += 1

    while queue:
        node = queue.popleft()
        new_left_node = TreeNode(val, left=node.left)
        new_right_node = TreeNode(val, right=node.right)

        node.left = new_left_node
        node.right = new_right_node
    return root


def add_one_row_one_go(root: TreeNode | None, val: int, depth: int) -> TreeNode | None:
    if not root:
        return None
    if depth == 1:
        return TreeNode(val, left=root)

    queue = deque([(root, 1)])
    while queue:
        node, curr_depth = queue.popleft()
        if curr_depth == depth - 1:
            new_left_node = TreeNode(val, left=node.left)
            new_right_node = TreeNode(val, right=node.right)

            node.left = new_left_node
            node.right = new_right_node
        else:
            if node.left:
                queue.append((node.left, curr_depth + 1))
            if node.right:
                queue.append((node.right, curr_depth + 1))
    return root


if __name__ == "__main__":
    r = TreeNode.deserialize([4, 2, 6, 3, 1, 5])
    print(r.serialize())
    print(add_one_row(r, val=1, depth=2).serialize())

    print(">" * 20)
    r = TreeNode.deserialize([4, 2, 6, 3, 1, 5])
    print(r.serialize())
    print(add_one_row(r, val=1, depth=1).serialize())

    print(">" * 20)
    r = TreeNode.deserialize([4, 2, None, 3, 1])
    print(r.serialize())
    print(add_one_row(r, val=1, depth=3).serialize())
