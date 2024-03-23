"""
https://leetcode.com/problems/reorder-list/description/

143. Reorder List
Medium

You are given the head of a singly linked-list. The list can be represented as:

L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.


Example 1:
Input: head = [1,2,3,4]
Output: [1,4,2,3]

Example 2:
Input: head = [1,2,3,4,5]
Output: [1,5,2,4,3]


Constraints:

The number of nodes in the list is in the range [1, 5 * 104].
1 <= Node.val <= 1000
"""

from collections import deque
from typing import Optional

from data_structures.lists import SingleLinkNode as ListNode


def reorder_list(head: Optional[ListNode]) -> None:
    """
    Do not return anything, modify head in-place instead.
    """

    if not head or not head.next or not head.next.next:
        # None, 1 or 2 elements only
        return

    # Populate the queue with all nodes to easily access them.
    queue = deque()
    curr = head
    while curr:
        queue.append(curr)
        curr = curr.next

    tail = None
    while len(queue) > 1:
        left = queue.popleft()  # Get the left-most (front) node.
        right = queue.pop()  # Get the right-most (back) node.

        if tail:
            # Link previous right to current left
            tail.next = left

        left.next = right  # Link current left to current right
        right.next = None
        tail = right  # Update tail to current right

    if queue:
        # Handle the last node in case of an odd number of nodes
        tail.next = queue[-1]
        queue[-1].next = None  # Ensure the last node points to None


def reorder_list_fast_slow_pointers(head: Optional[ListNode]) -> None:
    if not head or not head.next:
        return

    # Find the middle of the list
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    # Reverse the second half of the list
    prev, curr = None, slow.next
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp

    # Now prev is the head of the reversed second half
    # Reset slow.next to None to mark the end of the first half
    slow.next = None

    # Merge two halves
    first, second = head, prev
    while second:
        # Temporarily store next nodes
        temp1, temp2 = first.next, second.next

        # Reorder nodes
        first.next = second
        second.next = temp1

        # Move pointers forward
        first, second = temp1, temp2


def reorder_list_even_and_reversed_odd_indices(head: Optional[ListNode]) -> None:
    """
    You are given the head of a singly linked-list.
    The list can be represented as:
        L0 → L1 → … → Ln-1

    Reorder the list to be on the following form:
        L0 → Ln-1 → Ln-2 → L1, L2, → Ln-3 → Ln-4 → L3, → L4 … → Ln // 2

    You may not modify the values in the list's nodes.
    Only links may be changed.

    Note: implement a in-place solution modifying the head and do not return anything.
    """
    if not head or not head.next:
        # None or 1 element only
        return

    # else, there are more than 1 node in the list
    queue = deque()
    curr = head
    while curr:
        queue.append(curr)
        curr = curr.next

    tail = None
    flip = True
    while len(queue) > 1:
        flip ^= True
        left = queue.popleft()
        right = queue.pop()
        if flip:
            left, right = right, left

        if tail:
            # connect the previous right to the current left, if one was already visited
            tail.next = left

        left.next = right
        tail = right

    if queue:
        # handle odd sized lists
        tail.next = queue[0]
        queue[0].next = None
    else:
        tail.next = None
