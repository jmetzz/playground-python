import pytest
from conftest import TREE_0_VALUES, TREE_1_VALUES, TREE_2_VALUES
from data_structures import BinaryTreeNode
from pytest import fail


def test_binary_tree_build():
    fail("Test case not implemented yet")


def test_from_indices():
    fail("Test case not implemented yet")


def test_binary_tree_traverse_preorder():
    tree = BinaryTreeNode.build([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.preorder(tree) == [4, 2, 1, 3, 7, 6, 9]


def test_binary_tree_traverse_inorder():
    tree = BinaryTreeNode.build([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.inorder(tree) == [1, 2, 3, 4, 6, 7, 9]


def test_binary_tree_traverse_postorder():
    tree = BinaryTreeNode.build([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.postorder(tree) == [1, 3, 2, 6, 9, 7, 4]


@pytest.mark.parametrize(
    "input_tree_values, expected",
    [
        (TREE_0_VALUES, [1, 2, 3, 4, 5]),
        (TREE_1_VALUES, [4, 2, 7, 1, 3, 6, 9]),
        (TREE_2_VALUES, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_bfs_apply(input_tree_values, expected):
    accumulator = []
    BinaryTreeNode.build(input_tree_values).bfs_apply(lambda x: accumulator.append(x.val))
    assert accumulator == expected


@pytest.mark.parametrize(
    "input_tree_values, expected",
    [
        (TREE_0_VALUES, [1, 2, 3, 4, 5]),
        (TREE_1_VALUES, [4, 2, 7, 1, 3, 6, 9]),
        (TREE_2_VALUES, [1, 2, 3, 4, 5, 6]),
    ],
)
def test_binary_tree_bfs(input_tree_values, expected):
    assert BinaryTreeNode.build(input_tree_values).bfs() == expected


def test_binary_tree_invert():
    expected = [4, 7, 2, 9, 6, 3, 1]
    tree = BinaryTreeNode.build(TREE_1_VALUES)
    actual = tree.invert().bfs()
    assert actual == expected


def test_binary_tree_invert_recursive():
    expected = [4, 7, 2, 9, 6, 3, 1]
    tree = BinaryTreeNode.build(TREE_1_VALUES)
    actual = BinaryTreeNode.invert_recursive(tree).bfs()
    assert actual == expected
