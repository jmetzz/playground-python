import pytest
from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_623_add_one_row_to_tree import add_one_row, add_one_row_one_go


@pytest.mark.parametrize("func", [add_one_row, add_one_row_one_go])
@pytest.mark.parametrize(
    "input_tree_values, new_node_val, depth, expected",
    [
        ([4, 2, 6, 3, 1, 5], 1, 2, [4, 1, 1, 2, None, None, 6, 3, 1, 5]),
        ([4, 2, 6, 3, 1, 5], 1, 1, [1, 4, None, 2, 6, 3, 1, 5]),
        ([4, 2, None, 3, 1], 1, 3, [4, 2, None, 1, 1, 3, None, None, 1]),
    ],
)
def test_add_one_row(func, input_tree_values, new_node_val, depth, expected):
    root = TreeNode.deserialize(input_tree_values)
    actual = func(root, new_node_val, depth)
    assert actual.serialize() == expected
