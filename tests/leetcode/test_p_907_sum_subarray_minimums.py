import pytest
from leetcode.p_907_sum_subarray_minimums import (
    SumSubarrayMinsNaive,
    SumSubarrayMinsStackingA,
    SumSubarrayMinsStackingB,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 1, 2, 4], 17),
        ([11, 81, 94, 43, 3], 444),
    ],
)
def test_sum_subarray_min_naive(test_input, expected):
    assert expected == SumSubarrayMinsNaive().solve(test_input)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 1, 2, 4], 17),
        ([11, 81, 94, 43, 3], 444),
    ],
)
def test_sum_subarray_min_stacking_a(test_input, expected):
    assert expected == SumSubarrayMinsStackingA().solve(test_input)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([3, 1, 2, 4], 17),
        ([11, 81, 94, 43, 3], 444),
    ],
)
def test_sum_subarray_min_stacking_b(test_input, expected):
    assert expected == SumSubarrayMinsStackingB().solve(test_input)
