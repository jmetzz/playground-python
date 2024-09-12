import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_86_partition_list import partition_inplace


@pytest.mark.parametrize(
    "input_data, k, expected",
    [
        ([1, 4, 3, 2, 5, 2], 3, [1, 2, 2, 4, 3, 5]),
        ([2, 1], 2, [1, 2]),
    ],
)
def test_function_name(input_data, k, expected):
    head = ListNode.from_array(input_data)
    actual = partition_inplace(head, k)
    assert actual.as_array() == expected
