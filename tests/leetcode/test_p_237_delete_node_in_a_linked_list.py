import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_237_delete_node_in_a_linked_list import delete_node_1, delete_node_2, delete_node_3


@pytest.mark.parametrize("func", [delete_node_1, delete_node_2, delete_node_3])
@pytest.mark.parametrize(
    "input_arr, node_idx, expected_arr",
    [
        ([4, 5], 0, [5]),
        ([4, 5, 1, 9], 1, [4, 1, 9]),
        ([4, 5, 1, 9], 2, [4, 5, 9]),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 8, [0, 1, 2, 3, 4, 5, 6, 7, 9]),
    ],
)
def test_delete_node(func, input_arr, node_idx, expected_arr):
    head = ListNode.deserialize(input_arr)
    target_node = head
    for _ in range(node_idx):
        target_node = target_node.next

    func(target_node)  # func changes the list in-place

    assert head.as_array() == expected_arr
