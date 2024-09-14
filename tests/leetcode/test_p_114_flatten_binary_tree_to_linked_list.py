import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_114_flatten_binary_tree_to_linked_list import flatten


@pytest.mark.parametrize(
    "input_data, expected",
    [([1], [1]), ([1, 2, 5, 3, 4, None, 6], [1, None, 2, None, 3, None, 4, None, 5, None, 6])],
)
def test_flatten(input_data, expected):
    tree = TreeNode.deserialize(input_data)
    flatten(tree)
    assert tree.serialize() == expected
