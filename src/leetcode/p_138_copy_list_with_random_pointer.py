"""
https://leetcode.com/problems/copy-list-with-random-pointer/description

138. Copy List with Random Pointer

Medium

A linked list of length n is given such that each node contains an additional
random pointer, which could point to any node in the list, or null.

Construct a deep copy of the list. The deep copy should consist of exactly n
brand new nodes, where each new node has its value set to the value of its
corresponding original node. Both the next and random pointer of the new nodes
should point to new nodes in the copied list such that the pointers in the
original list and copied list represent the same list state.
None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list,
where X.random --> Y, then for the corresponding two nodes x and y in the
copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes.
Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random
pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.



Example 1:
Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

Example 2:
Input: head = [[1,1],[2,1]]
Output: [[1,1],[2,1]]

Example 3:
Input: head = [[3,null],[3,0],[3,null]]
Output: [[3,null],[3,0],[3,null]]


Constraints:

0 <= n <= 1000
-10^4 <= Node.val <= 10^4
Node.random is null or is pointing to some node in the linked list.

"""


class Node:
    def __init__(self, x: int, next_node: "Node" = None, random: "Node" = None):
        self.val = int(x)
        self.next = next_node
        self.random = random


def copy_random_list_two_pass_hashing(head: Node | None) -> Node | None:
    if not head:
        return None
    _map_old_to_new = {None: None}

    # first pass: clone the list nodes without the connections
    curr = head
    while curr:
        _map_old_to_new[curr] = Node(curr.val)
        curr = curr.next

    # second pass: fix the links
    curr = head
    while curr:
        clone = _map_old_to_new[curr]
        clone.next = _map_old_to_new[curr.next]
        clone.random = _map_old_to_new[curr.random]
        curr = curr.next

    return _map_old_to_new[head]


def copy_random_list_interweaving(head: Node | None) -> Node | None:
    if not head:
        return None

    # Step 1: Insert copied nodes
    curr = head
    while curr:
        new_node = Node(curr.val)
        new_node.next = curr.next
        curr.next = new_node
        curr = new_node.next

    # Step 2: Set random pointers for the copied nodes
    curr = head
    while curr:
        if curr.random:
            curr.next.random = curr.random.next
        curr = curr.next.next

    # Step 3: Separate the copied list from the original
    curr = head
    copy_head = head.next
    copy_curr = copy_head

    while curr:
        curr.next = curr.next.next
        if copy_curr.next:
            copy_curr.next = copy_curr.next.next
        curr = curr.next
        copy_curr = copy_curr.next

    return copy_head


def build_linked_list(values, random_indices) -> Node:
    """
    Helper function to create a linked list with `random` pointers.

    Args:
        values: A list of node values.
        random_indices: A list of indices where random[i] points to
                        the index random_indices[i].
                        If random_indices[i] is None, random pointer is None.

    Returns:
        The head of the created linked list.
    """
    if not values:
        return None

    nodes = [Node(val) for val in values]

    for i in range(len(values) - 1):
        nodes[i].next = nodes[i + 1]

    for i, random_index in enumerate(random_indices):
        if random_index is not None:
            nodes[i].random = nodes[random_index]

    return nodes[0]  # Return head of the linked list


def compare_linked_lists(head1, head2):
    """
    Compare two linked lists to check if they are structurally identical (but not the same object).

    Args:
        head1: The head of the first linked list.
        head2: The head of the second linked list.

    Returns:
        True if the lists are identical, False otherwise.
    """
    while head1 and head2:
        if head1 is head2:  # They should be different objects
            return False
        if head1.val != head2.val:
            return False
        if (head1.random and head2.random) and (head1.random.val != head2.random.val):
            return False
        if (head1.random is None) != (head2.random is None):
            return False
        head1 = head1.next
        head2 = head2.next
    return head1 is None and head2 is None  # Both should be None at the end


if __name__ == "__main__":
    head = build_linked_list([7, 13, 11, 10, 1], [None, 0, 4, 2, 0])
    copied_head_1 = copy_random_list_two_pass_hashing(head)
    copied_head_2 = copy_random_list_interweaving(head)

    assert head == copied_head_1
    assert head == copied_head_2
