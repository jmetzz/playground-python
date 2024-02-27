from conftest import TREE_1_VALUES
from data_structures import BinaryTreeNode
from leetcode.p_226_invert_binary_tree import invert_tree


def test_binary_tree_invert():
    expected = [4, 7, 2, 9, 6, 3, 1]
    tree = BinaryTreeNode.build(TREE_1_VALUES)
    actual = invert_tree(tree).bfs()
    assert actual == expected
