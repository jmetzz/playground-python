from data_structures.trees import BinaryTreeNode as TreeNode


def leaf_similar(root1: TreeNode | None, root2: TreeNode | None) -> bool:
    def _dfs(node, tree_id: int):
        if not node:
            return

        if node.left is None and node.right is None:
            tree_leaves_map[tree_id].append(node.val)
            return

        _dfs(node.left, tree_id)
        _dfs(node.right, tree_id)

    tree_leaves_map = {1: [], 2: []}
    _dfs(root1, 1)
    _dfs(root2, 2)
    return tree_leaves_map[1] == tree_leaves_map[2]


def leaf_similar_list(root1: TreeNode | None, root2: TreeNode | None) -> bool:
    def _dfs_get_leaf(root):
        if not root:
            return []
        if root.left is None and root.right is None:
            return [root.val]

        return _dfs_get_leaf(root.left) + _dfs_get_leaf(root.right)

    return _dfs_get_leaf(root1) == _dfs_get_leaf(root2)


if __name__ == "__main__":
    print(leaf_similar(TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9]), TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])))
    print(leaf_similar(TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9]), TreeNode.deserialize([4, 2, 7, 1, 3, 6])))

    print(leaf_similar_list(TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9]), TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])))
    print(leaf_similar_list(TreeNode.deserialize([4, 2, 7, 1, 3, 6, 9]), TreeNode.deserialize([4, 2, 7, 1, 3, 6])))
