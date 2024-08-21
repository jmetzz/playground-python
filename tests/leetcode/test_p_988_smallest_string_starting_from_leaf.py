import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_988_smallest_string_starting_from_leaf import smallest_from_leaf_bfs, smallest_from_leaf_dfs


@pytest.mark.parametrize("func", [smallest_from_leaf_dfs, smallest_from_leaf_bfs])
@pytest.mark.parametrize(
    "input_tree_values, expected",
    [([0, 1, 2, 3, 4, 3, 4], "dba"), ([25, 1, 3, 1, 3, 0, 2], "adz"), ([2, 2, 1, None, 1, 0, None, 0], "abc")],
)
def test_smallest_from_leaf(func, input_tree_values, expected):
    root = TreeNode.deserialize(input_tree_values)
    assert func(root) == expected
