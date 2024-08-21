import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_1161_max_level_sum_of_binary_tree import max_level_sum_1, max_level_sum_2

test_cases = [
    ([], 0),
    ([0], 1),
    ([9], 1),
    ([9, 1], 1),
    ([0, 1], 2),
    ([0, 1, 1], 2),
    ([1, 7, 0, 7, -8, None, None], 2),
    ([1, 2, 3, 4, 5, 6, 7, 8], 3),
    ([7, 7, -8, -7, None, 9], 1),
]


@pytest.mark.parametrize("input_arr, expected", test_cases)
def test_max_level_sum_1(input_arr, expected):
    root = TreeNode.deserialize(input_arr)
    assert max_level_sum_1(root) == expected


@pytest.mark.parametrize("input_arr, expected", test_cases)
def test_max_level_sum_2(input_arr, expected):
    root = TreeNode.deserialize(input_arr)
    assert max_level_sum_2(root) == expected
