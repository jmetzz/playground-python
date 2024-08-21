import pytest

from leetcode.p_2542_maximum_subsequence_score import max_score


@pytest.mark.parametrize(
    "nums1, nums2, k, expected", [([1, 3, 3, 2], [2, 1, 3, 4], 3, 12), ([4, 2, 3, 1, 1], [7, 5, 10, 9, 6], 1, 30)]
)
def test_max_score(nums1, nums2, k, expected):
    assert max_score(nums1, nums2, k) == expected
