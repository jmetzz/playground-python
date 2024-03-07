import pytest
from data_structures import SingleLinkListNode
from leetcode.p_876_middle_of_the_linked_list import middle_node, middle_node_two_pointers

INPUTS = [
    # (SingleLinkListNode.from_array([1]), 1),
    (SingleLinkListNode.from_array([]), None),
    # (SingleLinkListNode.from_array([1, 2, 3, 4, 5]), 3),
    # (SingleLinkListNode.from_array([1, 2, 3, 4, 5, 6]), 4),
]


@pytest.mark.parametrize("input_list, expected", INPUTS)
def test_middle_node(input_list, expected):
    actual_node = middle_node(input_list)
    if expected:
        assert actual_node.val == expected
    else:
        assert actual_node is None


@pytest.mark.parametrize("input_list, expected", INPUTS)
def test_middle_node_two_pointers(input_list, expected):
    actual_node = middle_node_two_pointers(input_list)
    if expected:
        assert actual_node.val == expected
    else:
        assert actual_node is None
