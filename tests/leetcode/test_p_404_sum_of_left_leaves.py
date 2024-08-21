import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_404_sum_of_left_leaves import sum_of_left_leaves


@pytest.mark.parametrize(
    "input_arr, expected",
    [([], 0), ([3], 0), ([3, 9, 20], 9), ([3, 9, 20, None, None, 15, 7], 24)],
)
def test_sum_of_left_leaves(input_arr, expected):
    assert sum_of_left_leaves(TreeNode.deserialize(input_arr)) == expected
