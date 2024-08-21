import pytest

from leetcode.p_345_reverse_vowels_of_string import reverse_vowels, reverse_vowels_2, reverse_vowels_stack


@pytest.mark.parametrize(
    "input_word, expected", [("hello", "holle"), ("leetcode", "leotcede"), ("race car", "race car")]
)
def test_reverse_vowels(input_word, expected):
    assert reverse_vowels(input_word) == expected


@pytest.mark.parametrize(
    "input_word, expected", [("hello", "holle"), ("leetcode", "leotcede"), ("race car", "race car")]
)
def test_reverse_vowels_2(input_word, expected):
    assert reverse_vowels_2(input_word) == expected


@pytest.mark.parametrize(
    "input_word, expected", [("hello", "holle"), ("leetcode", "leotcede"), ("race car", "race car")]
)
def test_reverse_vowels_stack(input_word, expected):
    assert reverse_vowels_stack(input_word) == expected
