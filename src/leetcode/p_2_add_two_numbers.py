"""
https://leetcode.com/problems/add-two-numbers/description

2. Add Two Numbers

Medium

You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order, and each of their nodes contains
a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except
the number 0 itself.


Example 1:
Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

Example 2:
Input: l1 = [0], l2 = [0]
Output: [0]

Example 3:
Input: l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
Output: [8,9,9,9,0,0,0,1]

Constraints:

The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading
zeros.
"""

from data_structures.lists import SingleLinkNode as ListNode


def add_two_numbers(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    if not l1:
        return l2
    if not l2:
        return l1

    p1, p2 = l1, l2
    head, tail = None, None
    carry = 0
    while p1 and p2:
        _sum = p1.val + p2.val + carry
        if _sum <= 9:
            value = _sum
            carry = 0
        else:
            value = _sum - 10
            carry = 1

        new_node = ListNode(value)
        if not head:
            tail = new_node
            head = tail
        else:
            tail.next = new_node
            tail = tail.next
        p1 = p1.next
        p2 = p2.next

    # still need to consider the carry value that might be 1
    if p1:
        while p1:
            _sum = p1.val + carry
            if _sum <= 9:
                value = _sum
                carry = 0
            else:
                value = _sum - 10
                carry = 1
            tail.next = ListNode(value)
            tail = tail.next
            p1 = p1.next
    if p2:
        while p2:
            _sum = p1.val + carry
            if _sum <= 9:
                value = _sum
                carry = 0
            else:
                value = _sum - 10
                carry = 1
            tail.next = ListNode(value)
            tail = tail.next
            p2 = p2.next
    if carry:
        tail.next = ListNode(carry)
    return head


def add_two_numbers_2(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    if not l1:
        return l2
    if not l2:
        return l1

    dummy_head = ListNode(None)
    tail = dummy_head
    p1, p2 = l1, l2
    carry = 0
    while p1 and p2:
        _sum = p1.val + p2.val + carry
        value, carry = _sum % 10, _sum // 10
        tail.next = ListNode(value)
        tail = tail.next
        p1 = p1.next
        p2 = p2.next

    # still need to consider the carry value that might be 1
    if p1:
        while p1:
            _sum = p1.val + carry
            value, carry = _sum % 10, _sum // 10
            tail.next = ListNode(value)
            tail = tail.next
            p1 = p1.next
    if p2:
        while p2:
            _sum = p2.val + carry
            value, carry = _sum % 10, _sum // 10
            tail.next = ListNode(value)
            tail = tail.next
            p2 = p2.next

    if carry:
        tail.next = ListNode(carry)
    return dummy_head.next


def add_two_numbers_3(l1: ListNode | None, l2: ListNode | None) -> ListNode | None:
    dummy_head = ListNode(None)
    tail = dummy_head
    p1, p2 = l1, l2  # using these to not change the input pointers
    carry = 0
    while p1 or p2 or carry != 0:
        p1_val = p1.val if p1 else 0
        p2_val = p2.val if p2 else 0
        _sum = p1_val + p2_val + carry
        value, carry = _sum % 10, _sum // 10
        tail.next = ListNode(value)
        tail = tail.next
        p1 = p1.next if p1 else None
        p2 = p2.next if p2 else None

    return dummy_head.next


if __name__ == "__main__":
    print(add_two_numbers_2(ListNode.from_array([0]), ListNode.from_array([0])).serialize())  # [0]
    a = add_two_numbers_2(ListNode.from_array([2, 4, 3]), ListNode.from_array([5, 6, 4]))  # [7,0,8]
    print(a.serialize())

    l1 = ListNode.from_array([9, 9, 9, 9, 9, 9, 9])
    l2 = ListNode.from_array([9, 9, 9, 9])
    l3 = add_two_numbers_2(l1, l2)  # 8,9,9,9,0,0,0,1]
    print(l3.serialize())

    print(add_two_numbers_3(ListNode.from_array([0]), ListNode.from_array([0])).serialize())  # [0]
    print(add_two_numbers_3(ListNode.from_array([2, 4, 3]), ListNode.from_array([5, 6, 4])).serialize())  # [7,0,8]

    l1 = ListNode.from_array([9, 9, 9, 9, 9, 9, 9])
    l2 = ListNode.from_array([9, 9, 9, 9])
    l3 = add_two_numbers_3(l1, l2)  # 8,9,9,9,0,0,0,1]
    print(l3.serialize())
