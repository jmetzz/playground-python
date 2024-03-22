from data_structures.lists import SingleLinkNode as ListNode
from hypothesis import given, settings, strategies
from leetcode.p_234_palindrome_linked_list import is_palindrome_list


def palindrome_list_strategy():
    return strategies.lists(strategies.integers(min_value=0, max_value=9)).map(lambda x: x + x[::-1])


@given(palindrome_list_strategy())
@settings(max_examples=10)
def test_is_palindrome_list(input_list_values):
    assert is_palindrome_list(ListNode.from_array(input_list_values)) is True


def non_palindrome_list_strategy(min_size=2, max_size=10):
    """
    Generates non-palindrome lists of integers.

    Args:
    - min_size: Minimum size of the list. Must be at least 2 to ensure non-palindromicity.
    - max_size: Maximum size of the list.

    Returns:
    - A strategy that generates non-palindrome lists of integers.
    """
    # Ensure the list has at least two elements to make non-palindromicity possible
    assert min_size >= 2, "Non-palindrome lists must contain at least 2 elements."

    def make_non_palindrome(x):
        if len(x) < 2:
            return x

        first_half, second_half = x[: len(x) // 2], x[len(x) // 2 :]
        if first_half[-1] == second_half[0]:
            # Increment the first element of the second half to ensure it's different,
            # wrapping around the value if it exceeds max_value.
            second_half[0] = (second_half[0] + 1) % 10
        return first_half + second_half

    return strategies.lists(strategies.integers(min_value=0, max_value=9), min_size=min_size, max_size=max_size).map(
        make_non_palindrome
    )


@given(non_palindrome_list_strategy(min_size=2, max_size=10))
@settings(max_examples=10)
def test_is_not_palindrome_list(input_list_values):
    assert is_palindrome_list(ListNode.from_array(input_list_values)) is False
