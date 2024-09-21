import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_103_binary_tree_zigzag_level_order_traversal import zigzag_level_order


@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([], []),
        ([1], [[1]]),
        ([3, 9, 20, None, None, 15, 7], [[3], [20, 9], [15, 7]]),
        ([1, 2, 3, 4, None, None, 5], [[1], [3, 2], [4, 5]]),
    ],
)
def test_zigzag_level_order(input_data, expected):
    tree = TreeNode.from_array(input_data)
    actual = zigzag_level_order(tree)
    assert actual == expected
