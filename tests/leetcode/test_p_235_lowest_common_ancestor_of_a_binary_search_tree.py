import pytest

from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_235_lowest_common_ancestor_of_a_binary_search_tree import lowest_common_ancestor_binary_search_tree


@pytest.mark.parametrize(
    ("arr_values", "p_value", "q_value", "expected"),
    [
        ([2, 1], 2, 1, 2),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 2, 8, 6),
        ([6, 2, 8, 0, 4, 7, 9, None, None, 3, 5], 2, 4, 2),
    ],
)
def test_lowest_common_ancestor_binary_search_tree(arr_values, p_value, q_value, expected):
    tree = TreeNode.from_array(arr_values)
    p_node = TreeNode.find_node_by_value(tree, p_value)
    q_node = TreeNode.find_node_by_value(tree, q_value)
    actual = lowest_common_ancestor_binary_search_tree(tree, p_node, q_node)
    assert actual.val == expected
