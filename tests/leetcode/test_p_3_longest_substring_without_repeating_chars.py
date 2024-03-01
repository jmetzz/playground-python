import pytest
from leetcode.p_3_longest_substring_without_repeating_chars import (
    length_of_longest_substring__dynamic_sliding_window_1,
    length_of_longest_substring__dynamic_sliding_window_2,
    length_of_longest_substring__fixed_sliding_window,
)


@pytest.mark.parametrize(
    "input_sequence, expected",
    [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("dvdf", 3),
        ("   ", 1),
        (" ", 1),
        ("", 0),
    ],
)
def test_fixed_sliding_window(input_sequence, expected):
    assert length_of_longest_substring__fixed_sliding_window(input_sequence) == expected


@pytest.mark.parametrize(
    "input_sequence, expected",
    [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("dvdf", 3),
        ("   ", 1),
        (" ", 1),
        ("", 0),
    ],
)
def test_dynamic_sliding_window_1(input_sequence, expected):
    assert length_of_longest_substring__dynamic_sliding_window_1(input_sequence) == expected


@pytest.mark.parametrize(
    "input_sequence, expected",
    [
        ("abcabcbb", 3),
        ("bbbbb", 1),
        ("pwwkew", 3),
        ("dvdf", 3),
        ("   ", 1),
        (" ", 1),
        ("", 0),
    ],
)
def test_dynamic_sliding_window_2(input_sequence, expected):
    assert length_of_longest_substring__dynamic_sliding_window_2(input_sequence) == expected
