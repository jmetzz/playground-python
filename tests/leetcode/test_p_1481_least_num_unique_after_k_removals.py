import pytest

from leetcode.p_1481_least_num_unique_after_k_removals import (
    FindLeastNumOfUniqueAfterRemoval,
)


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([5, 5, 4], 1, 1),
        ([4, 3, 1, 1, 3, 3, 2], 3, 2),
        ([2, 4, 1, 8, 3, 5, 1, 3], 3, 3),
    ],
)
def test_least_num_unique_after_removal_heap(test_input_arr, test_input_k, expected):
    assert FindLeastNumOfUniqueAfterRemoval().solve_with_heap(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([5, 5, 4], 1, 1),
        ([4, 3, 1, 1, 3, 3, 2], 3, 2),
        ([2, 4, 1, 8, 3, 5, 1, 3], 3, 3),
    ],
)
def test_least_num_unique_after_removal_array(test_input_arr, test_input_k, expected):
    assert FindLeastNumOfUniqueAfterRemoval().solve_with_array(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([5, 5, 4], 1, 1),
        ([4, 3, 1, 1, 3, 3, 2], 3, 2),
        ([2, 4, 1, 8, 3, 5, 1, 3], 3, 3),
    ],
)
def test_least_num_unique_after_removal_loop(test_input_arr, test_input_k, expected):
    assert FindLeastNumOfUniqueAfterRemoval().solve_with_one_loop(test_input_arr, test_input_k) == expected


@pytest.mark.parametrize(
    "test_input_arr, test_input_k, expected",
    [
        ([5, 5, 4], 1, 1),
        ([4, 3, 1, 1, 3, 3, 2], 3, 2),
        ([2, 4, 1, 8, 3, 5, 1, 3], 3, 3),
    ],
)
def test_least_num_unique_after_removal_ffreitas(test_input_arr, test_input_k, expected):
    assert FindLeastNumOfUniqueAfterRemoval().solve_ffreitas(test_input_arr, test_input_k) == expected
