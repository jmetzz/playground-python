import pytest

from leetcode.p_1456_max_num_vowels_in_substring_of_length import max_vowels


@pytest.mark.parametrize("word, window_size, expected", [("abciiidef", 3, 3), ("aeiou", 2, 2), ("leetcode", 3, 2)])
def test_max_vowels(word, window_size, expected):
    assert max_vowels(word, window_size) == expected
