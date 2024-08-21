import pytest

from leetcode.p_890_find_and_replace_pattern import find_and_replace_pattern


@pytest.mark.parametrize(
    "words, pattern, expected",
    [(["abc", "deq", "mee", "aqq", "dkd", "ccc"], "abb", ["mee", "aqq"]), (["a", "b", "c"], "a", ["a", "b", "c"])],
)
def test_find_and_replace_pattern(words, pattern, expected):
    assert find_and_replace_pattern(words, pattern) == expected
