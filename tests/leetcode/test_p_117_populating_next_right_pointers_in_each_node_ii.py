import pytest

from leetcode.p_117_populating_next_right_pointers_in_each_node_ii import Node, build_tree, connect, serialize_tree


def level_order_with_next_pointers(root: Node):
    """Helper function to retrieve the level order traversal using next pointers."""
    result = []
    curr = root
    while curr:
        level_nodes = []
        next_level_start = None
        while curr:
            level_nodes.append(curr.val)
            if not next_level_start:
                next_level_start = curr.left or curr.right
            curr = curr.next
        result.append(level_nodes)
        curr = next_level_start
    return result


@pytest.mark.parametrize(
    "tree_list, expected_output",
    [
        # Test case 1: Empty tree
        ([], ""),
        # Test case 2: Single node tree
        ([1], "1,#"),
        # Test case 3: Two-level tree
        ([1, 2, 3], "1,#,2,3,#"),
        # Test case 4: Three-level tree (perfect binary tree)
        ([1, 2, 3, 4, 5, 6, 7], "1,#,2,3,#,4,5,6,7,#"),
        # Test case 5: Left-skewed tree
        ([1, 2, None, 3], "1,#,2,#,3,#"),
        # Test case 6: Right-skewed tree
        ([1, None, 2, None, None, None, 3], "1,#,2,#,3,#"),
        # Test case 7: Mixed tree (uneven structure)
        ([1, 2, 3, 4, 5, None, 7], "1,#,2,3,#,4,5,7,#"),
        # Test case 8: Only right children
        ([1, None, 2, None, None, None, 3], "1,#,2,#,3,#"),
        # Test case 9: Only left children
        ([1, 2, None, 3], "1,#,2,#,3,#"),
    ],
)
def test_connect(tree_list, expected_output):
    root = build_tree(tree_list)
    connected_root = connect(root)
    assert serialize_tree(connected_root) == expected_output
