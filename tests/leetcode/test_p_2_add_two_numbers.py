import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_2_add_two_numbers import add_two_numbers, add_two_numbers_2, add_two_numbers_3


@pytest.mark.parametrize("func", [add_two_numbers, add_two_numbers_2, add_two_numbers_3])
@pytest.mark.parametrize(
    ("arr_1", "arr_2", "expected_arr"),
    [
        ([0], [0], [0]),
        ([2, 4, 3], [5, 6, 4], [7, 0, 8]),
        ([9, 9, 9, 9, 9, 9, 9], [9, 9, 9, 9], [8, 9, 9, 9, 0, 0, 0, 1]),
    ],
)
def test_add_two_numbers(func, arr_1, arr_2, expected_arr):
    l1 = ListNode.from_array(arr_1)
    l2 = ListNode.from_array(arr_2)
    actual = func(l1, l2)
    assert actual.as_array() == expected_arr
