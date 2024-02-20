from typing import List

import pytest
from leetcode.p_49_group_anagrams import GroupAnagrams


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ([""], [[""]]),
        (["a"], [["a"]]),
        (
            ["eat", "tea", "tan", "ate", "nat", "bat"],
            [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]],
        ),
    ],
)
def test_group_anagrams(test_input: List[str], expected: List[List[str]]):
    for e in expected:
        e.sort()

    actual = GroupAnagrams().solve(test_input)

    for group in actual:
        assert sorted(group) in expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        # ([""], [[""]]),
        # (["a"], [["a"]]),
        (
            ["eat", "tea", "tan", "ate", "nat", "bat"],
            [["bat"], ["nat", "tan"], ["ate", "eat", "tea"]],
        ),
    ],
)
def test_group_anagrams_with_custom_signature(test_input: List[str], expected: List[List[str]]):
    for e in expected:
        e.sort()

    actual = GroupAnagrams().solve_with_custom_signature(test_input)

    for group in actual:
        assert sorted(group) in expected
