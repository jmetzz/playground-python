import pytest
from leetcode.p_1768_merge_strings_alternately import merge_alternately


@pytest.mark.parametrize(
    "s1, s2, expected", [("abc", "pqr", "apbqcr"), ("ab", "pqrs", "apbqrs"), ("abcd", "pq", "apbqcd")]
)
def test_merge_alternatively(s1, s2, expected):
    assert merge_alternately(s1, s2) == expected
