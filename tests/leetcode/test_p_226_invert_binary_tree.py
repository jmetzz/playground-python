from conftest import TREE_1_VALUES
from data_structures import BinaryTreeNode
from leetcode.p_226_invert_binary_tree import invert_tree


def test_binary_tree_invert():
    expected = [0, 2, 1, 6, 5, 4, 3]
    tree = BinaryTreeNode.deserialize(TREE_1_VALUES)
    actual = invert_tree(tree).bfs()
    assert actual == expected
