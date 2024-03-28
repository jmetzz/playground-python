import pytest
from leetcode.p_1679_max_number_of_K_sum_pairs import max_operations_counter, max_operations_two_pointers


@pytest.mark.parametrize("function_under_test", [max_operations_two_pointers, max_operations_counter])
@pytest.mark.parametrize(
    "nums, k, expected",
    [
        ([], 5, 0),  # empty list
        ([1], 2, 0),  # single element
        ([1, 2, 3, 4], 5, 2),  # all elements can form pairs
        ([1, 5, 7, 1], 6, 1),  # only one pair can be formed
        ([4, 4, 1, 3, 1, 2, 2], 5, 3),  # mixed elements, some can form pairs
        ([3, 1, 3, 4, 3], 6, 1),  # duplicate elements that can form a single pair
    ],
)
def test_max_operations(function_under_test, nums, k, expected):
    assert function_under_test(nums, k) == expected
