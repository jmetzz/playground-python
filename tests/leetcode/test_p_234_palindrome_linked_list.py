import pytest
from data_structures.lists import SingleLinkNode as ListNode
from hypothesis import given, settings, strategies
from leetcode.p_234_palindrome_linked_list import is_palindrome_list


def palindrome_list_strategy():
    return strategies.lists(strategies.integers(min_value=0, max_value=9)).map(lambda x: x + x[::-1])


@given(palindrome_list_strategy())
@settings(max_examples=10)
def test_is_palindrome_list(input_list_values):
    assert is_palindrome_list(ListNode.from_array(input_list_values)) is True


non_palindrome_test_cases = [
    ([1, 2, 2, 4, 5, 6, 6]),
    ([1, 2]),
    ([1, 2, 3]),
    ([1, 2, 2, 3]),
    ([0, 0, 0, 0, 0, 1]),
]


@pytest.mark.parametrize("input_arr", non_palindrome_test_cases)
def test_is_not_palindrome_list(input_arr):
    assert is_palindrome_list(ListNode.from_array(input_arr)) is False
