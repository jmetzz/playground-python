import pytest

from leetcode.p_1673_find_the_most_competitive_subsequence import most_competitive_1, most_competitive_2


@pytest.mark.parametrize("func", [most_competitive_1, most_competitive_2])
@pytest.mark.parametrize(
    "input_arr, k, expected",
    [
        ([2, 4, 3, 3, 5, 4, 9, 6], 4, [2, 3, 3, 4]),
        ([71, 18, 52, 29, 55, 73, 24, 42, 66, 8, 80, 2], 3, [8, 80, 2]),
    ],
)
def test_most_competitive(func, input_arr, k, expected):
    assert func(input_arr, k) == expected
