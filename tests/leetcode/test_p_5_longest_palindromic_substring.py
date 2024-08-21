import pytest

from leetcode.p_5_longest_palindromic_substring import (
    backtrack_palindrome,
    longest_palindrome_brute_force,
    longest_palindrome_expanding_window,
    longest_palindrome_heap_brute_force,
)


@pytest.mark.parametrize("input_sequence, expected", [("cbbd", "bb"), ("babad", ["bab", "aba"]), ("abbbc", ["bbb"])])
def test_longest_palindrome_brute_force(input_sequence, expected):
    assert longest_palindrome_brute_force(input_sequence) in expected


@pytest.mark.parametrize("input_sequence, expected", [("cbbd", "bb"), ("babad", ["bab", "aba"]), ("abbbc", ["bbb"])])
def test_longest_palindrome_heap_brute_force(input_sequence, expected):
    assert longest_palindrome_heap_brute_force(input_sequence) in expected


@pytest.mark.parametrize("input_sequence, expected", [("cbbd", "bb"), ("babad", ["bab", "aba"]), ("abbbc", ["bbb"])])
def test_longest_palindrome_expanding_window(input_sequence, expected):
    assert longest_palindrome_expanding_window(input_sequence) in expected


@pytest.mark.parametrize(
    "input_sequence, center, expected",
    [
        ("cbbd", 0, "c"),
        ("cbbd", 1, "b"),
        ("cbbd", 2, "b"),
        ("babad", 0, "b"),
        ("babad", 1, "bab"),
        ("babad", 2, "aba"),
        ("babad", 3, "a"),
    ],
)
def test_backtrack_palindrome_odd_length(input_sequence, center, expected):
    assert backtrack_palindrome(input_sequence, center, center) == expected


@pytest.mark.parametrize(
    "input_sequence, center, expected",
    [
        ("cbbd", 0, ""),
        ("cbbd", 1, "bb"),
        ("cbbd", 2, ""),
        ("babad", 0, ""),
        ("babad", 1, ""),
        ("babad", 2, ""),
        ("babad", 3, ""),
        ("abbad", 0, ""),
        ("abbad", 1, "abba"),
        ("abbad", 2, ""),
    ],
)
def test_backtrack_palindrome_even_length(input_sequence, center, expected):
    assert backtrack_palindrome(input_sequence, center, center + 1) == expected
