import pytest

from leetcode.p_1383_maximum_performance_team import max_performance


@pytest.mark.parametrize(
    "nums1, nums2, k, expected",
    [
        ([2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 2, 60),
        ([2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 3, 68),
        ([2, 10, 3, 1, 5, 8], [5, 4, 3, 9, 7, 2], 4, 72),
        ([2, 8, 2], [2, 7, 1], 2, 56),
    ],
)
def test_max_performance(nums1, nums2, k, expected):
    assert max_performance(nums1, nums2, k) == expected
