import pytest
from leetcode.p_1457_pseudopalindromic_paths import (
    PseudoPalindromicPaths_iterative,
    PseudoPalindromicPaths_recursive,
    TreeNode,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([2, 3, 1, 3, 1, None, 1], 2),
        ([2, 1, 1, 1, 3, None, None, None, None, None, 1], 1),
        ([9], 1),
    ],
)
def test_pseudo_palindromic_paths_recursive(test_input, expected):
    tree = TreeNode.build(test_input, 0)
    assert PseudoPalindromicPaths_recursive().solve(tree) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([2, 3, 1, 3, 1, None, 1], 2),
        ([2, 1, 1, 1, 3, None, None, None, None, None, 1], 1),
        ([9], 1),
    ],
)
def test_pseudo_palindromic_paths_iterative(test_input, expected):
    tree = TreeNode.build(test_input, 0)
    assert PseudoPalindromicPaths_iterative().solve(tree) == expected
