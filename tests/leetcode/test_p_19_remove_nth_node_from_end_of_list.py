import pytest
from data_structures.lists import SingleLinkNode
from leetcode.p_19_remove_nth_node_from_end_of_list import remove_nth_from_end, remove_nth_from_end_two_pointers


@pytest.mark.parametrize(
    "input_arr, input_n, expected",
    [
        ([1, 2, 3, 4, 5], 2, [1, 2, 3, 5]),
        ([1], 1, None),
        ([1, 2], 1, [1]),
        ([1, 2], 2, [2]),
    ],
)
def test_remove_nth_from_end(input_arr, input_n, expected):
    head_node = SingleLinkNode.from_array(input_arr)
    actual = remove_nth_from_end(head_node, input_n)

    if expected:
        assert actual.as_array() == expected
    else:
        assert actual is None


@pytest.mark.parametrize(
    "input_arr, input_n, expected",
    [
        ([1, 2, 3, 4, 5], 2, [1, 2, 3, 5]),
        ([1], 1, None),
        ([1, 2], 1, [1]),
        ([1, 2], 2, [2]),
    ],
)
def test_remove_nth_from_end_with_two_pointers(input_arr, input_n, expected):
    head_node = SingleLinkNode.from_array(input_arr)
    actual = remove_nth_from_end_two_pointers(head_node, input_n)
    if expected:
        assert actual.as_array() == expected
    else:
        assert actual is None
