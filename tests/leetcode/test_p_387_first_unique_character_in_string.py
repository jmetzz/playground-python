import pytest
from leetcode.p_387_first_unique_character_in_string import (
    first_unique_char_1,
    first_unique_char_2,
    first_unique_char_3,
)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("leetcode", 0),
        ("loveleetcode", 2),
        ("aabb", -1),
        ("aabbc", 4),
        ("abracadabra", 4),
        ("abracadabrac", 6),
    ],
)
def test_first_unique_character_in_string_1(test_input, expected):
    assert first_unique_char_1(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("leetcode", 0),
        ("loveleetcode", 2),
        ("aabb", -1),
        ("aabbc", 4),
        ("abracadabra", 4),
        ("abracadabrac", 6),
    ],
)
def test_first_unique_character_in_string_2(test_input, expected):
    assert first_unique_char_2(test_input) == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("leetcode", 0),
        ("loveleetcode", 2),
        ("aabb", -1),
        ("aabbc", 4),
        ("abracadabra", 4),
        ("abracadabrac", 6),
    ],
)
def test_first_unique_character_in_string_3(test_input, expected):
    assert first_unique_char_3(test_input) == expected
