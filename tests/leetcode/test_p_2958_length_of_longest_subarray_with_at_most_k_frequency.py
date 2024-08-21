import pytest

from leetcode.p_2958_lenght_of_longest_subarray_with_at_most_k_frequency import max_subarray_length


@pytest.mark.parametrize(
    "input_arr, k, expected",
    [([1, 2, 3, 1, 2, 3, 1, 2], 2, 6), ([1, 2, 1, 2, 1, 2, 1, 2], 1, 2), ([5, 5, 5, 5, 5, 5, 5], 4, 4)],
)
def test_max_subarray_length(input_arr, k, expected):
    assert max_subarray_length(input_arr, k) == expected
