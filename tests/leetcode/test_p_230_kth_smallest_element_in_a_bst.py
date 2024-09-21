import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_230_kth_smallest_element_in_a_bst import (
    kth_smallest_iter_stack_double_loop,
    kth_smallest_iter_stack_single_loop,
    kth_smallest_recur_inoder,
    kth_smallest_recur_inoder_2,
)


@pytest.mark.parametrize(
    "func",
    [
        kth_smallest_recur_inoder,
        kth_smallest_recur_inoder_2,
        kth_smallest_iter_stack_double_loop,
        kth_smallest_iter_stack_single_loop,
    ],
)
@pytest.mark.parametrize(
    "input_data, k, expected", [([3, 1, 4, None, 2], 1, 1), ([5, 3, 6, 2, 4, None, None, 1], 3, 3)]
)
def test_kth_smallest(func, input_data, k, expected):
    tree = TreeNode.from_array([3, 1, 4, None, 2])
    assert func(tree, k) == expected
