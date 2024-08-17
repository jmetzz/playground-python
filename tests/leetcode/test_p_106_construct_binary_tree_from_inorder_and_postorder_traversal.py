import pytest
from leetcode.p_106_construct_binary_tree_from_inorder_and_postorder_traversal import build_tree, build_tree_with_helper


@pytest.mark.parametrize("func", [build_tree, build_tree_with_helper])
@pytest.mark.parametrize(
    "inorder, postorder, expected",
    [
        ([-1], [-1], [-1]),
        # ([9, 3, 15, 20, 7], [9, 15, 7, 20, 3], [3, 9, 20, None, None, 15, 7])
    ],
)
def test_build_tree(func, inorder, postorder, expected):
    tree = func(inorder, postorder)
    assert tree.serialize() == expected
