import pytest

from leetcode.p_49_group_anagrams import group_anagrams_solve, group_anagrams_solve_with_custom_signature


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
def test_group_anagrams(test_input: list[str], expected: list[list[str]]):
    for e in expected:
        e.sort()

    actual = group_anagrams_solve(test_input)

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
def test_group_anagrams_with_custom_signature(test_input: list[str], expected: list[list[str]]):
    for e in expected:
        e.sort()

    actual = group_anagrams_solve_with_custom_signature(test_input)

    for group in actual:
        assert sorted(group) in expected
