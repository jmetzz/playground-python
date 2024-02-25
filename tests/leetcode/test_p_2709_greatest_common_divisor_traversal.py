import pytest
from leetcode.p_2709_greatest_common_divisor_traversal import GreatestCommonDivisor


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([2, 3, 6], True),
        ([3, 9, 5], False),
        ([4, 3, 12, 8], True),
    ],
)
def test_greatest_common_divisor_(test_input, expected):
    assert GreatestCommonDivisor().solve_prime_factorization(test_input) == expected
