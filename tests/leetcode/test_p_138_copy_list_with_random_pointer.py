from leetcode.p_138_copy_list_with_random_pointer import (
    build_linked_list,
    compare_linked_lists,
    copy_random_list_interweaving,
    copy_random_list_two_pass_hashing,
)


def test_copy_random_list_two_pass_hashing():
    # Test case 1: Empty list
    assert copy_random_list_two_pass_hashing(None) is None

    # Test case 2: Single node without random pointer
    head = build_linked_list([1], [None])
    copied_head = copy_random_list_two_pass_hashing(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 3: Single node with random pointing to itself
    head = build_linked_list([1], [0])
    copied_head = copy_random_list_two_pass_hashing(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 4: Multiple nodes with no random pointers
    head = build_linked_list([1, 2, 3], [None, None, None])
    copied_head = copy_random_list_two_pass_hashing(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 5: Multiple nodes with random pointers
    head = build_linked_list([1, 2, 3], [2, 0, 1])  # Random: 1 -> 3, 2 -> 1, 3 -> 2
    copied_head = copy_random_list_two_pass_hashing(head)
    assert compare_linked_lists(copied_head, head)


def test_copy_random_list_interweaving():
    # Test case 1: Empty list
    assert copy_random_list_interweaving(None) is None

    # Test case 2: Single node without random pointer
    head = build_linked_list([1], [None])
    copied_head = copy_random_list_interweaving(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 3: Single node with random pointing to itself
    head = build_linked_list([1], [0])
    copied_head = copy_random_list_interweaving(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 4: Multiple nodes with no random pointers
    head = build_linked_list([1, 2, 3], [None, None, None])
    copied_head = copy_random_list_interweaving(head)
    assert compare_linked_lists(copied_head, head)

    # Test case 5: Multiple nodes with random pointers
    head = build_linked_list([1, 2, 3], [2, 0, 1])  # Random: 1 -> 3, 2 -> 1, 3 -> 2
    copied_head = copy_random_list_interweaving(head)
    assert compare_linked_lists(copied_head, head)
