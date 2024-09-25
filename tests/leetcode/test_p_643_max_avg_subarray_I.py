import pytest

from leetcode.p_643_max_avg_subarray_I import find_max_average_1, find_max_average_2


@pytest.mark.parametrize("func", [find_max_average_1, find_max_average_2])
@pytest.mark.parametrize(
    "nums, k, expected",
    [
        ([1, 12, -5, -6, 50, 3], 4, 12.75),
        ([5, 5, 5, 5, 5], 1, 5),
        ([5, 5, 5, 5, 5], 5, 5),
        ([-1, -2, -3, -4, -5], 2, -1.5),
        ([4, 0, 4, 3, 3], 5, 2.8),
        ([0, 4, 0, 3, 2], 1, 4),
        ([-1, -12, -5, -6, -50, -3], 4, -6),
    ],
)
def test_find_max_average(func, nums: list[int], k: int, expected: float):
    assert func(nums, k) == pytest.approx(expected, abs=1e-5)
