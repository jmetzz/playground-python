import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_92_reverse_linked_list_ii import reverse_between_1, reverse_between_2, reverse_between_3


@pytest.mark.parametrize("func", [reverse_between_1, reverse_between_2, reverse_between_3])
@pytest.mark.parametrize(
    ("arr_data", "left", "right", "expected"),
    [
        ([], 0, 0, []),
        ([5], 1, 1, [5]),
        ([1, 5], 1, 1, [1, 5]),
        ([1, 5], 2, 2, [1, 5]),
        ([1, 5], 1, 2, [5, 1]),
        ([1, 2, 3, 4, 5], 1, 5, [5, 4, 3, 2, 1]),
        ([1, 2, 3, 4, 5], 2, 4, [1, 4, 3, 2, 5]),
        ([1, 2, 3, 4, 5], 3, 3, [1, 2, 3, 4, 5]),
        ([1, 2, 3, 4, 5], 3, 5, [1, 2, 5, 4, 3]),
    ],
)
def test_reverse_linked_list_ii(func, arr_data, left, right, expected):
    head = ListNode.from_array(arr_data)
    rev_head = func(head, left, right)
    actual = rev_head.as_array() if rev_head else []
    assert actual == expected
