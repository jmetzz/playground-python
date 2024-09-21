import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_98_validate_binary_search_tree import is_valid_bst, is_valid_bst_2


@pytest.mark.parametrize("func", [is_valid_bst, is_valid_bst_2])
@pytest.mark.parametrize(
    "input_data, expected",
    [
        ([], False),
        ([1], True),
        ([2, 1], True),
        ([1, 2], False),
        ([1, 2, 3], False),
        ([2, 1, 3], True),
        ([5, 1, 4, None, None, 3, 6], False),
        ([5, 4, 6, None, None, 3, 7], False),
    ],
)
def test_is_valid_tree(func, input_data, expected):
    tree = TreeNode.from_array(input_data)
    assert func(tree) == expected
