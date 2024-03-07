import pytest
from leetcode.p_100_same_tree import is_same_tree
from trees import BinaryTreeNode


@pytest.mark.parametrize(
    "tree_1, tree_2, expected",
    [
        ([1, 2, 3], [1, 2, 3], True),
        ([1, 2], [1, None, 2], False),
        ([1, 2, 1], [1, 1, 2], False),
    ],
)
def test_is_same_tree(tree_1, tree_2, expected):
    t1 = BinaryTreeNode.deserialize(tree_1)
    t2 = BinaryTreeNode.deserialize(tree_2)
    assert is_same_tree(t1, t2) == expected
