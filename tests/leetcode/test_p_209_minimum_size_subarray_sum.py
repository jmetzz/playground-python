import pytest

from leetcode.p_209_minimum_size_subarray_sum import min_sub_array_len


@pytest.mark.parametrize(
    ("input_target", "input_nums", "expected"),
    [
        (5, [1, 1, 1], 0),
        (4, [1, 4, 4], 1),
        (7, [2, 3, 1, 2, 4, 3], 2),
        (15, [1, 2, 3, 4, 5], 5),
    ],
)
def test_min_sub_array_len(input_target, input_nums, expected):
    assert min_sub_array_len(input_target, input_nums) == expected
