import pytest

from data_structures.lists import SingleLinkNode
from leetcode.p_1669_merge_in_between_linked_lists import merge_in_between


@pytest.mark.parametrize(
    "list1, a, b, list2, expected",
    [
        ([], 0, 0, [1000], [1000]),
        ([1], 0, 0, [2000, 3000], [2000, 3000]),
        ([1, 2], 0, 0, [2000, 3000], [2000, 3000, 2]),
        ([1, 2], 1, 1, [2000, 3000], [1, 2000, 3000]),
        ([1, 2, 3], 1, 1, [2000, 3000], [1, 2000, 3000, 3]),
        ([10, 1, 13, 6, 9, 5], 3, 4, [1000000, 1000001, 1000002], [10, 1, 13, 1000000, 1000001, 1000002, 5]),
        (
            [0, 1, 2, 3, 4, 5, 6],
            2,
            5,
            [1000000, 1000001, 1000002, 1000003, 1000004],
            [0, 1, 1000000, 1000001, 1000002, 1000003, 1000004, 6],
        ),
    ],
)
def test_merge_in_between(list1, a, b, list2, expected):
    l1 = SingleLinkNode.deserialize(list1)
    l2 = SingleLinkNode.deserialize(list2)
    actual = merge_in_between(l1, a, b, l2)
    actual_array = SingleLinkNode.as_array(actual)
    assert actual_array == expected
