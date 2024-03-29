"""
https://leetcode.com/problems/odd-even-linked-list/description/

328. Odd Even Linked List
Medium

Given the head of a singly linked list, group all the nodes with odd indices
together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain
as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time
complexity.

Example 1:
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]

Example 2:
Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]


Constraints:

The number of nodes in the linked list is in the range [0, 104].
-106 <= Node.val <= 106
"""

from typing import Optional

from data_structures.lists import SingleLinkNode as ListNode


def odd_even_list_flag_flipping(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head

    odd = head
    even_list_head = even = head.next

    curr_node = even.next
    is_odd = False
    while curr_node:
        is_odd ^= True
        if is_odd:
            odd.next = curr_node
            odd = odd.next  # advance the odd list tail
        else:
            even.next = curr_node
            even = even.next  # advance the even list tail
        curr_node = curr_node.next  # advance the current node

    # connect left side with right side
    odd.next = even_list_head
    even.next = None
    return head


def odd_even_list_skipping_two(head: Optional[ListNode]) -> Optional[ListNode]:
    if not head or not head.next:
        return head
    odd = head
    even_list_head = even = head.next

    # [1, 2, 3, 4, 5]
    while even and even.next:
        odd.next = odd.next.next
        even.next = even.next.next

        odd = odd.next
        even = even.next

    odd.next = even_list_head
    return head
