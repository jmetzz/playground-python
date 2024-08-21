import pytest

from leetcode.p_67_add_binary import add_binary, add_binary_cheeting


@pytest.mark.parametrize("func", [add_binary, add_binary_cheeting])
@pytest.mark.parametrize(
    "input_str_a, input_str_b, expected", [("1010", "1011", "10101"), ("1", "111", "1000"), ("111", "1", "1000")]
)
def test_add_binary_same_length(func, input_str_a, input_str_b, expected):
    assert func(input_str_a, input_str_b) == expected
