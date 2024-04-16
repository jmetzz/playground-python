import pytest
from conftest import (
    TREE_0_VALUES,
    TREE_1_VALUES,
    TREE_2_VALUES,
)
from data_structures.trees import BinaryTreeNode


@pytest.mark.parametrize(
    "tree_values, tree_indices",
    [
        ([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]),
    ],
)
def test_binary_tree_creation(tree_values, tree_indices):
    t1 = BinaryTreeNode.deserialize(tree_values)
    t2 = BinaryTreeNode.from_indices(tree_indices)
    assert BinaryTreeNode.is_equal(t1, t2)


def test_binary_tree_traverse_preorder():
    tree = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.preorder(tree) == [4, 2, 1, 3, 7, 6, 9]


def test_binary_tree_traverse_inorder():
    tree = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.inorder(tree) == [1, 2, 3, 4, 6, 7, 9]


def test_binary_tree_traverse_postorder():
    tree = BinaryTreeNode.deserialize([4, 2, 7, 1, 3, 6, 9])
    assert BinaryTreeNode.postorder(tree) == [1, 3, 2, 6, 9, 7, 4]


@pytest.mark.parametrize(
    "input_tree_values, expected",
    [
        (TREE_0_VALUES, [0, 1, 2, 3, 4]),
        (TREE_1_VALUES, [0, 1, 2, 3, 4, 5, 6]),
        (TREE_2_VALUES, [0, 1, 3, 4, 7, 10]),
    ],
)
def test_bfs_apply(input_tree_values, expected):
    accumulator = []
    BinaryTreeNode.deserialize(input_tree_values).bfs_apply(lambda x: accumulator.append(x.val))
    assert accumulator == expected


@pytest.mark.parametrize(
    "input_tree_values, expected",
    [
        (TREE_0_VALUES, [0, 1, 2, 3, 4]),
        (TREE_1_VALUES, [0, 1, 2, 3, 4, 5, 6]),
        (TREE_2_VALUES, [0, 1, 3, 4, 7, 10]),
    ],
)
def test_binary_tree_bfs(input_tree_values, expected):
    assert BinaryTreeNode.deserialize(input_tree_values).bfs() == expected


def test_binary_tree_invert():
    expected = [0, 2, 1, 6, 5, 4, 3]
    tree = BinaryTreeNode.deserialize(TREE_1_VALUES)
    actual = tree.invert().bfs()
    assert actual == expected


def test_binary_tree_invert_recursive():
    expected = [0, 2, 1, 6, 5, 4, 3]
    tree = BinaryTreeNode.deserialize(TREE_1_VALUES)
    actual = BinaryTreeNode.invert_recursive(tree).bfs()
    assert actual == expected


def test_serialize():
    serialized_tree = [1, 2, 3, None, None, 4, 5]
    new_tree = BinaryTreeNode.deserialize(serialized_tree)
    assert new_tree.serialize() == serialized_tree
