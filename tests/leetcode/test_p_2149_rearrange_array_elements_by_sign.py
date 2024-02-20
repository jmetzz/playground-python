import pytest
from leetcode.p_2149_rearrange_array_elements_by_sign import RearrangeBySign


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 1, -2, -5, 2, -4], [3, -2, 1, -5, 2, -4]), ([-1, 1], [1, -1])],
)
def test_sequential_digits_zip(test_input, expected):
    assert RearrangeBySign().solve_with_zip(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 1, -2, -5, 2, -4], [3, -2, 1, -5, 2, -4]), ([-1, 1], [1, -1])],
)
def test_sequential_digits_arrays(test_input, expected):
    assert RearrangeBySign().solve_with_arrays(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 1, -2, -5, 2, -4], [3, -2, 1, -5, 2, -4]), ([-1, 1], [1, -1])],
)
def test_sequential_digits_side_effect(test_input, expected):
    assert RearrangeBySign().solve_with_side_effect(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [([3, 1, -2, -5, 2, -4], [3, -2, 1, -5, 2, -4]), ([-1, 1], [1, -1])],
)
def test_sequential_digits_deque(test_input, expected):
    assert RearrangeBySign().solve_with_deque(test_input) == expected
