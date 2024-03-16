import pytest
from data_structures.trees import BinaryTreeNode as TreeNode
from leetcode.p_101_symmetric_tree import is_symmetric_iterative, is_symmetric_recursive


@pytest.mark.parametrize("func", [is_symmetric_recursive, is_symmetric_iterative])
@pytest.mark.parametrize(
    "tree_data,expected",
    [
        ([], True),  # Empty tree
        ([1], True),  # Single node tree
        ([1, 2, 2, 3, 4, 4, 3], True),  # Complete symmetric tree
        ([1, 2, 3], False),
        ([1, 2, 2, None, 3, None, 3], False),  # Simple asymmetric tree  # Asymmetric tree
        ([1, 2, 2, 3, None, None, 3, 4, None, None, 4], False),  # Symmetric tree with missing nodes
    ],
)
def test_is_symmetric(func, tree_data, expected):
    root = TreeNode.deserialize(tree_data)
    assert func(root) == expected
