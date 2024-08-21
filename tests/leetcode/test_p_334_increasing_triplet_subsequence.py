import pytest

from leetcode.p_334_increasing_triplet_subsequence import increasing_triplet


@pytest.mark.parametrize(
    "input_arr, expected",
    [
        ([1, 2, 3, 4, 5], True),
        ([5, 4, 3, 2, 1], False),
        ([2, 1, 5, 0, 4, 6], True),
        ([20, 100, 10, 12, 5, 13], True),
    ],
)
def test_increasing_triplet(input_arr, expected):
    assert increasing_triplet(input_arr) == expected
