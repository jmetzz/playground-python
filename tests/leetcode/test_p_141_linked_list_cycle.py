from typing import List

import pytest
from data_structures.lists import SingleLinkNode
from leetcode.p_141_linked_list_cycle import has_cycle


def factory_list(arr: List[int], cycle_to_idx: int) -> SingleLinkNode:
    h = SingleLinkNode.from_array(arr)
    if cycle_to_idx < 0 or cycle_to_idx >= len(arr):
        # no cycle
        return h

    target_node = h
    for _ in range(cycle_to_idx):
        target_node = target_node.next
    tail = h
    while tail.next:
        tail = tail.next
    tail.next = target_node
    return h


@pytest.mark.parametrize(
    "list_values,cycle_to_idx, expected",
    [
        ([1], 0, True),
        ([1, 2], 0, True),
        ([3, 2, 0, -4], 1, True),
        ([1], -1, False),
        ([], -1, False),
        ([1], 1, False),
    ],
)
def test_has_cycle(list_values, cycle_to_idx, expected):
    head = factory_list(list_values, cycle_to_idx)
    assert has_cycle(head) == expected
