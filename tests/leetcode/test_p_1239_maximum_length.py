import pytest

from leetcode.p_1239_maximum_length import MaxLengthStrWithUniqueChars


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (["aa", "bb"], 0),
        (["un", "iq", "ue"], 4),
        (["cha", "r", "act", "ers"], 6),
        (["abcdefghijklmnopqrstuvwxyz"], 26),
        (["a", "abc", "d", "de", "def"], 6),
    ],
)
def test_max_length_str_with_unique_chars(test_input, expected):
    assert expected == MaxLengthStrWithUniqueChars().solve(test_input)
