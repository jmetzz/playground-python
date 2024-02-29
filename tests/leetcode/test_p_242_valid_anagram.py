import pytest
from leetcode.p_242_valid_anagram import (
    is_anagram_256array,
    is_anagram_builtin_count,
    is_anagram_counter,
    is_anagram_sorting,
)


@pytest.mark.parametrize("this, other, expected", [("anagram", "nagaram", True), ("rat", "car", False)])
def test_is_anagram_sorting(this, other, expected):
    assert is_anagram_sorting(this, other) == expected


@pytest.mark.parametrize("this, other, expected", [("anagram", "nagaram", True), ("rat", "car", False)])
def test_is_anagram_counter(this, other, expected):
    assert is_anagram_counter(this, other) == expected


@pytest.mark.parametrize("this, other, expected", [("anagram", "nagaram", True), ("rat", "car", False)])
def test_is_anagram_256array(this, other, expected):
    assert is_anagram_256array(this, other) == expected


@pytest.mark.parametrize("this, other, expected", [("anagram", "nagaram", True), ("rat", "car", False)])
def test_is_anagram_builtin_count(this, other, expected):
    assert is_anagram_builtin_count(this, other) == expected
