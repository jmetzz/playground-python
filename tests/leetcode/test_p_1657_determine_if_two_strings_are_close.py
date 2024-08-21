import pytest

from leetcode.p_1657_determine_if_two_strings_are_close import close_strings_array, close_strings_counter


@pytest.mark.parametrize(
    "word1, word2, expected", [("abc", "bca", True), ("a", "aa", False), ("cabbba", "abbccc", True)]
)
def test_close_strings_array(word1, word2, expected):
    assert close_strings_array(word1, word2) == expected


@pytest.mark.parametrize(
    "word1, word2, expected", [("abc", "bca", True), ("a", "aa", False), ("cabbba", "abbccc", True)]
)
def test_close_strings_counter(word1, word2, expected):
    assert close_strings_counter(word1, word2) == expected
