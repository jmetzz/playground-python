import pytest
from leetcode.p_2000_reverse_prefix_of_word import reverse_prefix


@pytest.mark.parametrize(
    "word, letter, expected", [("abcdefd", "d", "dcbaefd"), ("abcd", "z", "abcd"), ("xyxzxe", "z", "zxyxxe")]
)
def test_reverse_prefix(word, letter, expected):
    assert reverse_prefix(word, letter) == expected
