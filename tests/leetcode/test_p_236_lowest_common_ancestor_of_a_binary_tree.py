import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_236_lowest_common_ancestor_of_a_binary_tree import lowest_common_ancestor_generic_binary_tree


@pytest.mark.parametrize(
    ("arr_values", "p_value", "q_value", "expected"),
    [
        ([1, 2], 1, 2, 1),
        ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 1, 3),
        ([3, 5, 1, 6, 2, 0, 8, None, None, 7, 4], 5, 4, 5),
    ],
)
def test_lowest_common_ancestor_generic_binary_tree(arr_values, p_value, q_value, expected):
    tree = TreeNode.from_array(arr_values)
    p_node = TreeNode.find_node_by_value(tree, p_value)
    q_node = TreeNode.find_node_by_value(tree, q_value)
    actual = lowest_common_ancestor_generic_binary_tree(tree, p_node, q_node)
    assert actual.val == expected
