import pytest
from leetcode.p_394_decode_string import decode_string_nested, decode_string_non_nested


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("3[a]2[bc]", "aaabcbc"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef"),
        ("10[leetcode]", "leetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcode"),
    ],
)
def test_decode_string_non_nested(input_str, expected):
    assert decode_string_non_nested(input_str) == expected


@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("3[a]2[bc]", "aaabcbc"),
        ("2[abc]3[cd]ef", "abcabccdcdcdef"),
        ("10[leetcode]", "leetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcodeleetcode"),
        ("3[a2[c]]", "accaccacc"),
        ("2[abc]3[c5[d]]ef", "abcabccdddddcdddddcdddddef"),
        ("3[z]2[2[y]pq4[2[jk]e1[f]]]ef", "zzzyypqjkjkefjkjkefjkjkefjkjkefyypqjkjkefjkjkefjkjkefjkjkefef"),
    ],
)
def test_decode_string_nested(input_str, expected):
    assert decode_string_nested(input_str) == expected
