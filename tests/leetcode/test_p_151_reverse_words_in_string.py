import pytest
from leetcode.p_151_reverse_words_in_string import (
    reverse_words_iterative,
    reverse_words_iterative_2,
    reverse_words_recursive,
    reverse_words_using_builtin_split,
)

TEST_PARAMETERS = [
    ("the sky is blue", "blue is sky the"),
    ("  hello world  ", "world hello"),
    ("  hello    world  ", "world hello"),
    ("a good   example", "example good a"),
    ("a", "a"),
    ("     a", "a"),
    ("a     ", "a"),
    ("    a     ", "a"),
    ("", ""),
    ("     ", ""),
]


@pytest.mark.parametrize("input_string, expected", TEST_PARAMETERS)
def test_reverse_words_recursive(input_string, expected):
    assert reverse_words_recursive(input_string) == expected


@pytest.mark.parametrize("input_string, expected", TEST_PARAMETERS)
def test_reverse_words_iterative(input_string, expected):
    assert reverse_words_iterative(input_string) == expected


@pytest.mark.parametrize("input_string, expected", TEST_PARAMETERS)
def test_reverse_words_iterative_2(input_string, expected):
    assert reverse_words_iterative_2(input_string) == expected


@pytest.mark.parametrize("input_string, expected", TEST_PARAMETERS)
def test_reverse_words_builtin_split(input_string, expected):
    assert reverse_words_using_builtin_split(input_string) == expected
