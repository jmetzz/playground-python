import pytest

from leetcode.p_125_valid_palindrome import is_sentence_palindrome_1, is_sentence_palindrome_2, is_sentence_palindrome_3


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
    ],
)
def test_is_palindrome_from_phrase_1(input_phrase, expected):
    assert is_sentence_palindrome_1(input_phrase) == expected


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
    ],
)
def test_is_palindrome_from_phrase_2(input_phrase, expected):
    assert is_sentence_palindrome_2(input_phrase) == expected


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
    ],
)
def test_is_palindrome_from_phrase_3(input_phrase, expected):
    assert is_sentence_palindrome_3(input_phrase) == expected
