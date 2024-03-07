import pytest
from leetcode.p_543_diameter_of_binary_tree import (
    dfs_height,
    dfs_height_memo,
    diameter_of_bin_tree_bottom_up,
    diameter_of_bin_tree_memoization,
    diameter_of_bin_tree_recursive,
)
from trees import BinaryTreeNode

TREE_1 = BinaryTreeNode.deserialize([1, 2, 3, 4, 5])
TREE_2 = BinaryTreeNode.deserialize([1, 2, None, 3, 4, None, None, 5, None, None, 6])
TREE_3 = BinaryTreeNode.from_indices([0, 1, 2, 5, 6, 11, 13, 14, 23, 24, 29])
TREE_4 = BinaryTreeNode.from_indices([0, 1, 2, 5, 6, 11, 12, 13, 23, 25, 26, 47, 51, 53, 95, 96, 107, 192, 193])


@pytest.mark.parametrize(
    "tree, expected",
    [
        (TREE_1, 3),
        (TREE_2, 4),
        (TREE_3, 6),
        (TREE_4, 9),
    ],
)
def test_diameter_of_bin_tree_recursive(tree, expected):
    assert diameter_of_bin_tree_recursive(tree) == expected


@pytest.mark.parametrize(
    "tree, expected",
    [
        (TREE_1, 3),
        (TREE_2, 4),
        (TREE_3, 6),
        (TREE_4, 9),
    ],
)
def test_diameter_of_bin_tree_memoization(tree, expected):
    assert diameter_of_bin_tree_memoization(tree) == expected


@pytest.mark.parametrize(
    "tree, expected",
    [
        (TREE_1, 3),
        (TREE_2, 4),
        (TREE_3, 6),
        (TREE_4, 9),
    ],
)
def test_diameter_of_bin_tree_bottom_up(tree, expected):
    assert diameter_of_bin_tree_bottom_up(tree) == expected


@pytest.mark.parametrize(
    "tree, expected",
    [
        (TREE_1, 2),
        (TREE_2, 3),
        (TREE_3, 4),
        (TREE_4, 7),
    ],
)
def test_dfs_height(tree, expected):
    assert dfs_height(tree) == expected


@pytest.mark.parametrize(
    "tree, expected",
    [
        (TREE_1, 2),
        (TREE_2, 3),
        (TREE_3, 4),
        (TREE_4, 7),
    ],
)
def test_dfs_height_memo(tree, expected):
    assert dfs_height_memo(tree, {}) == expected
