import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_1171_remove_zero_sum_consecutive_nodes_from_list import remove_zero_sum_sublists


@pytest.mark.parametrize(
    "input_list,expected",
    [
        ([], []),  # Empty list
        ([1, 2, 3], [1, 2, 3]),  # No zero-sum sublists
        ([1, -1], []),  # Entire list sums to zero
        ([0], []),  # Single node that is zero
        ([1, 2, -3, 3, 1], [3, 1]),
        ([1, 2, -3, 3, -2, 4, -4, 5], [3, -2, 5]),  # Multiple non-overlapping zero-sum sublists
        ([1, 2, 3, -3, -2], [1]),  # Zero-sum sublist at the end
        ([-1, 1, -1, 1], []),  # Overlapping zero-sum sublists
        ([3, 4, -7, 5, -6, 6], [5]),
    ],
)
def test_remove_zero_sum_sublists(input_list, expected):
    head = ListNode.deserialize(input_list)
    actual = remove_zero_sum_sublists(head)
    if expected:
        assert actual.as_array() == expected, f"Failed for input {input_list}"
    else:
        assert actual is None
