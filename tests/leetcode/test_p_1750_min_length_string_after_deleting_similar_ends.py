import pytest

from leetcode.p_1750_min_length_string_after_deleting_similar_ends import minimum_length_loop, minimum_length_recursive


@pytest.mark.parametrize(
    "input_sequence, expected",
    [
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbbcbccbbabbb", 0),
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb", 1),
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb", 1),
        ("abbbbbbbbbbbbbbbccbcbcbccbba", 1),
        ("bbbbbbbbbbbbbbbccbcbcbccbb", 1),
        ("ccbcbcbcc", 1),
        ("bcbcb", 1),
        ("cbc", 1),
        ("b", 1),
        ("bb", 0),
        ("ca", 2),
        ("cabaabac", 0),
        ("aabccabba", 3),
        ("abcabcbac", 9),
    ],
)
def test_minimum_length(input_sequence, expected):
    assert minimum_length_recursive(input_sequence) == expected


@pytest.mark.parametrize(
    "input_sequence, expected",
    [
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbbcbccbbabbb", 0),
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb", 1),
        ("bbbbbbbbbbbbbbbbbbbbbbbbbbbabbbbbbbbbbbbbbbccbcbcbccbbabbb", 1),
        ("abbbbbbbbbbbbbbbccbcbcbccbba", 1),
        ("bbbbbbbbbbbbbbbccbcbcbccbb", 1),
        ("ccbcbcbcc", 1),
        ("bcbcb", 1),
        ("cbc", 1),
        ("b", 1),
        ("bb", 0),
        ("ca", 2),
        ("cabaabac", 0),
        ("aabccabba", 3),
        ("abcabcbac", 9),
    ],
)
def test_minimum_length_loop(input_sequence, expected):
    assert minimum_length_loop(input_sequence) == expected
