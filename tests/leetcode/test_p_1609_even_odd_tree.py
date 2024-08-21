import pytest

from data_structures.trees import BinaryTreeNode
from leetcode.p_1609_even_odd_tree import is_even_odd_tree


@pytest.mark.parametrize(
    "input_tree, expected",
    [
        ([1, 10, 4, 3, None, 7, 9, 12, 8, None, None, 6, None, None, 2], True),
        ([5, 4, 2, 3, 3, 7], False),
        ([5, 9, 1, 3, 5, 7], False),
        (
            [11, 18, 14, 3, 7, None, None, None, None, 18, None, None, None, None, None, None, None, None, None, 6],
            False,
        ),
        ([1, 10, 4, 3, None, 7, 9, 12, 8, 6, None, None, 2], True),
    ],
)
def test_is_even_odd_tree(input_tree, expected):
    tree = BinaryTreeNode.deserialize(input_tree)
    assert is_even_odd_tree(tree) == expected
