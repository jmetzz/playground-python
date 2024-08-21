import pytest

from leetcode.p_105_construct_binary_tree_from_preorder_and_inorder_traversal import build_tree, build_tree_with_helper


@pytest.mark.parametrize("func", [build_tree, build_tree_with_helper])
@pytest.mark.parametrize(
    "preorder, inorder, expected",
    [([-1], [-1], [-1]), ([3, 9, 20, 15, 7], [9, 3, 15, 20, 7], [3, 9, 20, None, None, 15, 7])],
)
def test_build_tree(func, preorder, inorder, expected):
    tree = func(preorder, inorder)
    assert tree.serialize() == expected
