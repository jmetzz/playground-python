import pytest
from leetcode.p_1493_longest_subarray_after_deleting_one_element import longest_subarray_sliding_window


@pytest.mark.parametrize("input_arr, expected", [([1, 1, 0, 1], 3), ([1, 1, 1], 2), ([0, 1, 1, 1, 0, 1, 1, 0, 1], 5)])
def test_longest_subarray_sliding_window(input_arr, expected):
    assert longest_subarray_sliding_window(input_arr) == expected
