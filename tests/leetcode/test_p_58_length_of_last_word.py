import pytest
from leetcode.p_58_length_of_last_word import (
    length_of_last_word,
    length_of_last_word_cheating_1,
    length_of_last_word_cheating_2,
)


@pytest.mark.parametrize("func", [length_of_last_word_cheating_1, length_of_last_word_cheating_2, length_of_last_word])
@pytest.mark.parametrize(
    "input_text, expected",
    [(" ", 0), ("  bla  ", 3), ("Hello World", 5), ("   fly me   to   the moon  ", 4), ("luffy is still joyboy", 6)],
)
def test_length_of_last_word(func, input_text, expected):
    assert func(input_text) == expected
