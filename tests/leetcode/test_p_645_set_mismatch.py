import pytest

from leetcode.p_645_set_mismatch import SetMismatch


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([1, 2, 2, 4], [2, 3]),
        ([1, 1], [1, 2]),
        ([2, 2], [2, 1]),
        ([1, 2, 3, 4, 9, 6, 7, 8, 9], [9, 5]),
        ([4, 8, 1, 5, 2, 7, 4, 6], [4, 3]),
    ],
)
def test_set_mismatch_setandsum(test_input, expected):
    assert SetMismatch().solve_with_setandsum(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([1, 2, 2, 4], [2, 3]),
        ([1, 1], [1, 2]),
        ([2, 2], [2, 1]),
        ([1, 2, 3, 4, 9, 6, 7, 8, 9], [9, 5]),
        ([4, 8, 1, 5, 2, 7, 4, 6], [4, 3]),
    ],
)
def test_set_mismatch_setandsum_2(test_input, expected):
    assert SetMismatch().solve_with_setandsum_2(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([1, 2, 2, 4], [2, 3]),
        ([1, 1], [1, 2]),
        ([2, 2], [2, 1]),
        ([1, 2, 3, 4, 9, 6, 7, 8, 9], [9, 5]),
        ([4, 8, 1, 5, 2, 7, 4, 6], [4, 3]),
    ],
)
def test_set_mismatch_with_maps(test_input, expected):
    assert SetMismatch().solve_with_maps(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([1, 2, 2, 4], [2, 3]),
        ([1, 1], [1, 2]),
        ([2, 2], [2, 1]),
        ([1, 2, 3, 4, 9, 6, 7, 8, 9], [9, 5]),
        ([4, 8, 1, 5, 2, 7, 4, 6], [4, 3]),
    ],
)
def test_set_mismatch_with_xor(test_input, expected):
    assert SetMismatch().solve_with_xor_ops(test_input) == expected
