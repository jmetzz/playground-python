import pytest

from algorithms.strings.palindromes import (
    is_palindrome_1,
    is_palindrome_2,
    is_palindrome_expanding_window,
    is_palindrome_with_stack,
)


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("amanaplanacanalpanama", True),
        ("raceacar", False),
        ("racecar", True),
        ("abba", True),
        ("pebcak", False),
        ("coffee", False),
        ("a", True),
        (" ", True),
    ],
)
def test_is_palindrome_stack_solution(input_phrase, expected):
    assert is_palindrome_with_stack(input_phrase, {}) == expected


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("amanaplanacanalpanama", True),
        ("raceacar", False),
        ("racecar", True),
        ("abba", True),
        ("pebcak", False),
        ("coffee", False),
        ("a", True),
        (" ", True),
    ],
)
def test_is_palindrome_solution_1(input_phrase, expected):
    assert is_palindrome_expanding_window(input_phrase) == expected


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("amanaplanacanalpanama", True),
        ("raceacar", False),
        ("racecar", True),
        ("abba", True),
        ("pebcak", False),
        ("coffee", False),
        ("a", True),
        (" ", True),
    ],
)
def test_is_palindrome_solution_2(input_phrase, expected):
    assert is_palindrome_1(input_phrase) == expected


@pytest.mark.parametrize(
    "input_phrase, expected",
    [
        ("amanaplanacanalpanama", True),
        ("raceacar", False),
        ("racecar", True),
        ("abba", True),
        ("pebcak", False),
        ("coffee", False),
        ("a", True),
        (" ", True),
    ],
)
def test_is_palindrome_solution_3(input_phrase, expected):
    assert is_palindrome_2(input_phrase) == expected
