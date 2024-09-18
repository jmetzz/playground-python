import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_102_binary_tree_level_order_traversal import level_order


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([], []),
        ([1], [[1]]),
        ([3, 9, 20, None, None, 15, 7], [[3], [9, 20], [15, 7]]),
    ],
)
def test_level_order_traversal(input_data, expected):
    tree = TreeNode.from_array(input_data)
    assert level_order(tree) == expected
