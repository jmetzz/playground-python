import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_61_rotate_list import rotate_right


@pytest.mark.parametrize(
    "input_data, k, expected",
    [
        ([1], 4, [1]),
        ([1, 2, 3, 4, 5], 7, [4, 5, 1, 2, 3]),
        ([1, 2, 3, 4, 5], 2, [4, 5, 1, 2, 3]),
        ([0, 1, 2], 4, [2, 0, 1]),
    ],
)
def test_function_name(input_data, k, expected):
    head = ListNode.from_array(input_data)
    actual = rotate_right(head, k)
    assert actual.as_array() == expected
