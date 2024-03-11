import pytest
from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_2095_delete_middle_node_linked_list import delete_middle


@pytest.mark.parametrize(
    "input_list, expected",
    [
        ([], None),
        ([1], None),
        ([2, 3], [2]),
        ([1, 2, 3], [1, 3]),
        ([1, 2, 3, 4], [1, 2, 4]),
        ([1, 3, 4, 7, 1, 2, 6], [1, 3, 4, 1, 2, 6]),
    ],
)
def test_delete_middle(input_list, expected):
    h = delete_middle(ListNode.from_array(input_list))
    if expected:
        assert h.asarray() == expected
    else:
        assert h is None
