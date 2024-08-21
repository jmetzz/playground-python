import pytest

from leetcode.p_1291_sequential_digits import SequentialDigits


@pytest.mark.parametrize(
    "test_input_low, test_input_high, expected",
    [
        (100, 300, [123, 234]),
        (1000, 13000, [1234, 2345, 3456, 4567, 5678, 6789, 12345]),
    ],
)
def test_sequential_digits(test_input_low, test_input_high, expected):
    assert SequentialDigits().solve(test_input_low, test_input_high) == expected
