import pytest
from leetcode.p_647_palindromic_substrings import PalindromicSubstrings


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("", True),
        ("a", True),
        ("aa", True),
        ("aaa", True),
        ("aaaa", True),
        ("aba", True),
        ("ababa", True),
        ("ababac", False),
        ("abcbac", False),
        ("palindrome", False),
    ],
)
def test_is_palindrome(test_input, expected):
    assert PalindromicSubstrings.is_palindrome(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("a", 1),  # 1
        ("aa", 3),  # 3
        ("aaa", 6),  # 6
        ("aaaa", 10),  # 10
    ],
)
def test_count_palindromic_substr_expanding_window(test_input, expected):
    assert PalindromicSubstrings().count_palindromic_substr_expanding_window(test_input) == expected  # 1


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("a", 1),  # 1
        ("aa", 3),  # 3
        ("aaa", 6),  # 6
        ("aaaa", 10),  # 10
    ],
)
def test_count_palindromic_substrings_loop(test_input, expected):
    assert PalindromicSubstrings().count_palindromic_substrings_loop(test_input) == expected  # 1
