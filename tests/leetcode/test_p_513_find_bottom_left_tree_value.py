import pytest
from conftest import TREE_0, TREE_1, TREE_2, TREE_3, TREE_4, TREE_5, TREE_6
from leetcode.p_513_find_bottom_left_tree_value import (
    find_bottom_left_value_iterative,
    find_bottom_left_value_recursive,
)


@pytest.mark.parametrize(
    "input_tree, expected",
    [
        (TREE_0, 4),
        (TREE_1, 1),
        (TREE_2, 5),
        (TREE_3, 23),
        (TREE_4, 192),
        (TREE_5, 7),
        (TREE_6, 11),
    ],
)
def test_find_bottom_left_value_iterative(input_tree, expected):
    assert find_bottom_left_value_iterative(input_tree) == expected


@pytest.mark.parametrize(
    "input_tree, expected",
    [(TREE_0, 4), (TREE_1, 1), (TREE_2, 5), (TREE_3, 23), (TREE_4, 192), (TREE_5, 7), (TREE_6, 11)],
)
def test_find_bottom_left_value_recursive(input_tree, expected):
    assert find_bottom_left_value_recursive(input_tree) == expected
