import pytest
from leetcode.p_1544_make_the_string_great import make_good


@pytest.mark.parametrize("input_word, expected", [("leEeetcode", "leetcode"), ("abBAcC", ""), ("s", "s")])
def test_make_good(input_word, expected):
    assert make_good(input_word) == expected
