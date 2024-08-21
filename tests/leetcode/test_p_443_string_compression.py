import pytest

from leetcode.p_443_string_compression import compress


@pytest.mark.parametrize(
    "input_chars, expected_chars, expected",
    [
        (["a", "a", "b", "b", "c", "c", "c"], ["a", "2", "b", "2", "c", "3"], 6),
        (["a"], ["a"], 1),
        (["a", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b", "b"], ["a", "b", "1", "2"], 4),
        (["a", "a", "a", "b", "b", "a", "a"], ["a", "3", "b", "2", "a", "2"], 6),
    ],
)
def test_compress(input_chars, expected_chars, expected):
    assert compress(input_chars) == expected
    assert input_chars[:expected] == expected_chars
