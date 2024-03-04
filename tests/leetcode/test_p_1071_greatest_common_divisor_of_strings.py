import pytest
from leetcode.p_1071_greatest_common_divisor_of_strings import gcd_of_strings


@pytest.mark.parametrize(
    "word1, word2, expected",
    [
        ("ABCABC", "ABC", "ABC"),
        ("ABABAB", "ABAB", "AB"),
        ("LEET", "CODE", ""),
    ],
)
def test_gcd_of_strings(word1, word2, expected):
    assert gcd_of_strings(word1, word2) == expected
