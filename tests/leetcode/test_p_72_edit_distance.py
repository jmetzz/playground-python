import pytest

from leetcode.p_72_edit_distance import edit_distance, edit_distance_recursive


@pytest.mark.parametrize(
    "word1, word2, expected",
    [
        ("abd", "acd", 1),
        ("abd", "acd", 1),
    ],
)
def test_edit_distance(word1, word2, expected):
    assert edit_distance(word1, word2) == expected


@pytest.mark.parametrize(
    "word1, word2, expected",
    [
        ("abd", "acd", 1),
        ("abd", "acd", 1),
    ],
)
def test_edit_distance_recursive(word1: str, word2: str, expected):
    assert edit_distance_recursive(word1, word2) == expected
