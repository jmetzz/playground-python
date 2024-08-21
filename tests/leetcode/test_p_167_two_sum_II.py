import pytest

from leetcode.p_167_two_sum_II import two_sum_bisect, two_sum_brute_force, two_sum_twopointers


@pytest.mark.parametrize("func", [two_sum_brute_force, two_sum_bisect, two_sum_twopointers])
@pytest.mark.parametrize(
    "input_numbers, target, expected",
    [
        ([-1, 0], -1, [1, 2]),
        ([2, 3, 4], 6, [1, 3]),
        ([2, 7, 11, 15], 9, [1, 2]),
        ([1, 3, 4, 5, 7, 10, 11], 9, [3, 4]),
    ],
)
def test_function_name(func, input_numbers, target, expected):
    assert func(input_numbers, target) == expected
