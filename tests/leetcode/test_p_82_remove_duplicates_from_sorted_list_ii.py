import pytest

from data_structures.lists import SingleLinkNode as ListNode
from leetcode.p_82_remove_duplicates_from_sorted_list_ii import (
    delete_duplicates_inplace_fast_slow,
    delete_duplicates_naive,
    delete_duplicates_one_pass,
)


@pytest.mark.parametrize(
    "func",
    [delete_duplicates_naive, delete_duplicates_one_pass, delete_duplicates_inplace_fast_slow],
)
@pytest.mark.parametrize(
    "arr_data, expected",
    [([1], [1]), ([1, 2], [1, 2]), ([1, 2, 3, 3, 4, 4, 5], [1, 2, 5]), ([1, 1, 1, 2, 3], [2, 3])],
)
def test_remove_duplicates(func, arr_data, expected):
    head = ListNode.from_array(arr_data)
    actual = func(head)
    assert actual.as_array() == expected


@pytest.mark.parametrize(
    "func",
    [delete_duplicates_naive, delete_duplicates_one_pass, delete_duplicates_inplace_fast_slow],
)
def test_remove_duplicates_empty_list(func):
    assert func(None) is None
