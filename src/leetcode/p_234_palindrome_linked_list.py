"""https://leetcode.com/problems/palindrome-linked-list

234. Palindrome Linked List
Easy

Given the head of a singly linked list, return true if it is a
palindrome or false otherwise.


Example 1:
Input: head = [1,2,2,1]
Output: true

Example 2:
Input: head = [1,2]
Output: false

Constraints:
The number of nodes in the list is in the range [1, 105].
0 <= Node.val <= 9

Follow up: Could you do it in O(n) time and O(1) space?
"""

from data_structures.lists import SingleLinkNode as ListNode


def is_palindrome_list(head: ListNode | None) -> bool:
    if not head:
        return True

    dummy = ListNode(next_node=head)
    slow, fast = head, head
    # traverse the list with two pointers, a fast and a slow pointer.
    # At the same time, creating a reversed list of the first half
    # of the original list, so that we can traverse it in reverse later.
    # slow represent the head of the inverted list
    head_2 = dummy
    steps = 0
    while fast and fast.next:
        steps += 1
        fast = fast.next.next
        temp = slow.next
        slow.next = head_2
        head_2 = slow
        slow = temp
    dummy.next = None

    # here slow should be at the middle of the list
    # we can iterate left and right at the same time
    # from slow (the middle) comparing the list elements
    # however, the behaviour differs slightly if the
    # the original list size is even or odd
    size = steps * 2 if fast is None else steps * 2 + 1
    if size == 1:
        return True

    if size % 2 != 0:
        slow = slow.next

    while head_2 and slow:
        if head_2.val != slow.val:
            return False
        head_2 = head_2.next
        slow = slow.next
    return True
