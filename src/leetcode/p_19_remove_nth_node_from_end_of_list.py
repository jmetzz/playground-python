"""
https://leetcode.com/problems/remove-nth-node-from-end-of-list/description

19. Remove Nth Node From End of List
Medium

Given the head of a linked list, remove the nth node from the end of the list and return its head.



Example 1:
Input: head = [1,2,3,4,5], n = 2
Output: [1,2,3,5]

Example 2:
Input: head = [1], n = 1
Output: []

Example 3:
Input: head = [1,2], n = 1
Output: [1]


Constraints:
The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

Follow up: Could you do this in one pass?
"""
from typing import Optional

from data_structures import SingleLinkNode


def remove_nth_from_end(head: Optional[SingleLinkNode], n: int) -> Optional[SingleLinkNode]:
    pointer = head
    stack = []
    while pointer:
        stack.append(pointer)
        pointer = pointer.next

    idx = len(stack) - n
    if idx == 0:
        # we need to remove the head of the current list
        new_head = head.next
        head.next = None  # clean up the connection to the list
        return new_head
    else:
        # stack[idx - 1] is the preceding node
        stack[idx - 1].next = stack[idx].next
        # clean up connection from the removed node.
        stack[idx].next = None
        # the head of the list stays the same
        return head


def remove_nth_from_end_two_pointers(head: Optional[SingleLinkNode], n: int) -> Optional[SingleLinkNode]:
    """
    Use the two pointers and dummy node technique.

    [1 ---> 2 ---> 3 ---> 4 ---> 5]
    n = 2, makes node 4 the target for removal

    We move left and right pointers keeping the distance
    between them at 2 (ie, n), while stopping the left pointer
    at the node just before the target.

    dummy --> 1 ---> 2 ---> 3 ---> 4 ---> 5 .
                            ^                   ^
                            L ----------------- R

    Args:
        head: the node at the first position
        n: 1 <= n <= size of the list

    Returns:
        the head of the list

    """
    # keep the distance between two pointers (left and right) equals to n.
    # insert a dummy node at start of the list to guarantee the left pointer
    # stop at the node preceding the target node for removal

    dummy_node = SingleLinkNode(None, head)
    # advance the right pointer n steps
    right_p = head
    steps = 0
    while steps < n and right_p:
        right_p = right_p.next
        steps += 1

    # mode both left and right pointers to the right until
    # the right point is out of the list, while keeping the distance
    # between them at n
    left_p = dummy_node
    while right_p:
        left_p = left_p.next
        right_p = right_p.next

    # remove the targe node, which is pointed
    # by left.next
    target = left_p.next  # keep this reference for clean up later
    left_p.next = target.next
    target.next = None  # clear this connection to avoid dangling pointers
    # return the new head
    return dummy_node.next


if __name__ == "__main__":
    inputs = [
        ([1, 2, 3, 4, 5], 2),
        ([1], 1),
        ([1, 2], 1),
        ([1, 2], 2),
    ]
    for arr, m in inputs:
        h = SingleLinkNode.from_array(arr)
        print(h.serialize() if h else [])
        # h = remove_nth_from_end(h, m)
        h = remove_nth_from_end_two_pointers(h, m)
        print(h.serialize() if h else [])
        print()
