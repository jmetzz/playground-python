"""
https://leetcode.com/problems/reverse-linked-list-ii

92. Reverse Linked List II

Medium
Given the head of a singly linked list and two integers left and right
where left <= right, reverse the nodes of the list from position left
to position right, and return the reversed list.

Example 1:
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]

Example 2:
Input: head = [5], left = 1, right = 1
Output: [5]


Constraints:

The number of nodes in the list is n.
1 <= n <= 500
-500 <= Node.val <= 500
1 <= left <= right <= n


Follow up: Could you do it in one pass?
"""

from data_structures.lists import SingleLinkNode as ListNode


def reverse_between_1(head: ListNode | None, left: int, right: int) -> ListNode | None:
    if not head or left >= right:
        return head

    dummy = ListNode(0, head)
    left_prev, cur = dummy, head

    # find the start of the sublist we want to reverse
    for _ in range(left - 1):
        left_prev = left_prev.next
        cur = cur.next

    # At this point:
    # left_prev -> the node just before the left-th node
    # cur -> the left-th node
    # reverse the sublist up to 'right' index.
    prev = None
    for _ in range(right - left + 1):
        temp = cur.next  # Save the next pointer
        cur.next = prev
        prev = cur
        cur = temp

    # After reversing:
    # prev -> the new head of the reversed sublist
    # cur -> the node after the 'right-th' node

    # readjust the pointers:
    # curr points to the "right sublist head"
    left_prev.next.next = cur
    # prev points to "the former right most node in the target sublist",
    # which now becomes head of the reversed sublist
    left_prev.next = prev

    return dummy.next


def reverse_between_2(head: ListNode | None, left: int, right: int) -> ListNode | None:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)
    left_prev, cur = dummy, head

    # Find the start of the sublist we want to reverse
    for _ in range(left - 1):
        left_prev = left_prev.next
        cur = cur.next

    # Reverse the sublist up to 'right' index
    prev = None
    tail_of_reversed_sublist = cur  # This will eventually point to the node after 'right'

    for _ in range(right - left + 1):
        temp = cur.next  # Save the next pointer
        cur.next = prev
        prev = cur
        cur = temp

    # Reconnect the reversed sublist with the rest of the list
    left_prev.next = prev  # Connect the node before 'left' to the new head of the reversed sublist
    tail_of_reversed_sublist.next = cur  # Connect the original 'left' node (now the tail) to the node after 'right'

    return dummy.next


def reverse_between_3(head: ListNode | None, left: int, right: int) -> ListNode | None:
    if not head or left == right:
        return head

    dummy = ListNode(0, head)  # Create a dummy node to simplify edge cases
    before_left = dummy

    # Move `before_left` to just before the `left-th` node
    for _ in range(left - 1):
        before_left = before_left.next

    # `curr` is the `left-th` node, which will become the tail of the reversed sublist
    curr = before_left.next

    # Reverse the sublist in-place using pointer manipulation
    # Key idea:
    # * The variable curr refers to the first node in the sublist that we're reversing.
    #   Initially, this is the node at position left. As we reverse nodes,
    #   curr stays in place, and nodes are "pulled" in front of it
    #   to reverse the order.
    # * The variable before_left refers to the node just before the sublist.
    #   It stays fixed during the reversal and is used to reattach the
    #   reversed nodes to the earlier part of the list.
    for _ in range(right - left):
        temp = curr.next  # the node we want to "pull" to the front in this iteration
        curr.next = temp.next  # detach temp from its current position by bypassing it
        temp.next = before_left.next  # insert temp at the front of the sublist
        before_left.next = temp  # we update before_left.next making temp the new head of the reversed of the sublist

    return dummy.next


if __name__ == "__main__":
    h = ListNode.from_array([1, 2, 3, 4, 5])
    print(h.serialize())

    h_r = reverse_between_1(h, 2, 4)
    print(h_r.serialize())

    h = ListNode.from_array([1, 2, 3, 4, 5])
    print(reverse_between_1(h, 1, 3).serialize())
