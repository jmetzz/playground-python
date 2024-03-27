import pytest
from leetcode.p_28_needle_in_haystack import str_in_str__cheeting, str_in_str_bruteforce


@pytest.mark.parametrize("func", [str_in_str_bruteforce, str_in_str__cheeting])
@pytest.mark.parametrize(
    "haystack, needle, expected",
    [("sadbutsad", "sad", 0), ("leetcode", "leeto", -1)],
)
def test_need_in_the_haystack(func, haystack, needle, expected):
    assert func(haystack, needle) == expected
