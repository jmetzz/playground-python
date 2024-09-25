"""https://leetcode.com/problems/reverse-linked-list/description

206. Reverse Linked List
Solved
Easy
Topics
Companies
Given the head of a singly linked list, reverse the list, and return the reversed list.


Example 1:
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]

Example 2:
Input: head = [1,2]
Output: [2,1]

Example 3:
Input: head = []
Output: []

Constraints:
The number of nodes in the list is the range [0, 5000].
-5000 <= Node.val <= 5000

Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?
"""

from data_structures.lists import SingleLinkNode as ListNode


def reverse_list_with_stack(head: ListNode | None) -> ListNode | None:
    """Reverses a linked list using a stack data structure.

    Args:
    ----
        head (Optional[ListNode]): The head of the linked list to be reversed.

    Returns:
    -------
        Optional[ListNode]: The new head of the reversed linked list.

    """
    if not head:
        return None

    stack = []
    curr = head
    while curr:
        stack.append(curr)
        curr = curr.next

    new_head = stack.pop()
    curr = new_head
    while stack:
        curr.next = stack.pop()
        curr = curr.next
    curr.next = None

    return new_head


def reverse_list_with_pointers(head: ListNode | None) -> ListNode | None:
    """Reverses a singly linked list using a pointer approach.

    The function iterates through the list, reversing the direction of the pointers as it goes.
    It keeps track of three nodes at a time: the previous node, the current node, and the next node.
    At each iteration, it changes the next pointer of the current node to point to the previous node,
    effectively reversing the link. Then it moves the previous and current pointers one step forward.
    The process continues until the end of the list is reached, at which point the previous node
    (now the last node of the original list) becomes the new head of the reversed list.

    Args:
    ----
        head: The head node of the singly linked list to be reversed.

    Returns:
    -------
        The new head node of the reversed linked list. If the input list is empty, returns None.

    """
    if not head:
        return None

    prev_node, curr_node = None, head
    while curr_node:
        next_node = curr_node.next  # Save the next node
        curr_node.next = prev_node  # Reverse the link
        prev_node = curr_node  # Move prev_node to current position
        curr_node = next_node  # Move to the next node in the list
    return prev_node  # The new head of the reversed list
