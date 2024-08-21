import pytest

from leetcode.p_1915_number_wonderful_substrings import wonderful_substrings


@pytest.mark.parametrize("word, expected", [("aba", 4), ("aabb", 9), ("he", 2)])
def test_wonderful_substrings(word, expected):
    assert wonderful_substrings(word) == expected
