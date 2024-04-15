import pytest
from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_129_sum_root_to_leaf_numbers import sum_root_to_leaf_path


@pytest.mark.parametrize("input_arr, expected", [([1, 2, 3], 25), ([4, 9, 0, 5, 1], 1026)])
def test_sum_root_to_leaf_path(input_arr, expected):
    assert sum_root_to_leaf_path(TreeNode.deserialize(input_arr)) == expected
