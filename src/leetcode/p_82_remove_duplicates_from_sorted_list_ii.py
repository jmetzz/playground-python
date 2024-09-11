"""
https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/

82. Remove Duplicates from Sorted List II
Medium

Given the head of a sorted linked list, delete all nodes that have duplicate
numbers, leaving only distinct numbers from the original list.
Return the linked list sorted as well.


Example 1:
Input: head = [1,2,3,3,4,4,5]
Output: [1,2,5]

Example 2:
Input: head = [1,1,1,2,3]
Output: [2,3]


Constraints:

The number of nodes in the list is in the range [0, 300].
-100 <= Node.val <= 100
The list is guaranteed to be sorted in ascending order.

"""

from collections import defaultdict

from data_structures.lists import SingleLinkNode as ListNode


def delete_duplicates_naive(head: ListNode | None) -> ListNode | None:
    if not head:
        return None

    curr = head
    counter = defaultdict(int)
    while curr:
        counter[curr.val] += 1
        curr = curr.next
    dummy_head = ListNode(None)

    # since the default behavior is to retrieve in order of insertion
    # in the dict ...
    tail = dummy_head
    for val, count in counter.items():
        if count > 1:
            # skip elements with more than 1 note in the original list
            continue
        # create a new node at the tail of the resulting list
        tail.next = ListNode(val)
        tail = tail.next
    return dummy_head.next


def delete_duplicates_one_pass(head: ListNode | None) -> ListNode | None:
    if not head:
        return None

    dummy_head = ListNode(None)
    curr, count, tail = head, 1, dummy_head
    while curr.next:
        if curr.val == curr.next.val:
            count += 1
            curr = curr.next
            continue

        if count == 1:
            # add curr_node to the resulting list
            tail.next = ListNode(curr.val)
            tail = tail.next

        count = 1  # reset counter
        curr = curr.next
    if count == 1:
        tail.next = ListNode(curr.val)
    return dummy_head.next


def delete_duplicates_inplace_fast_slow(head: ListNode | None) -> ListNode | None:
    if not head:
        return None

    dummy_head = ListNode(None, head)  # Dummy node to handle edge cases
    slow, fast = dummy_head, head

    while fast:
        # Check if the current node is part of a duplicate sequence
        if fast.next and fast.val == fast.next.val:
            # Skip all nodes with the same value as fast
            while fast.next and fast.val == fast.next.val:
                fast = fast.next
            # Link slow.next to the node after the duplicates
            slow.next = fast.next
        else:
            # Move slow to the next node only if no duplicate was found
            slow = slow.next
        # Move fast to the next node
        fast = fast.next

    return dummy_head.next


if __name__ == "__main__":
    head = ListNode.from_array([1, 2, 3, 3, 4, 4, 5])
    res = delete_duplicates_one_pass(head)
    print(res.serialize())  # [1, 2, 5]

    head = ListNode.from_array([1, 1, 1, 2, 3])
    res = delete_duplicates_one_pass(head)
    print(res.serialize())  # [2, 3]
