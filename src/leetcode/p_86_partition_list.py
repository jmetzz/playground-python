"""
https://leetcode.com/problems/partition-list/description
86. Partition List

Medium
Given the head of a linked list and a value x, partition it such
that all nodes less than x come before nodes greater than or equal to x.

You should preserve the original relative order of the nodes in each
 of the two partitions.



Example 1:
Input: head = [1,4,3,2,5,2], x = 3
Output: [1,2,2,4,3,5]

Example 2:
Input: head = [2,1], x = 2
Output: [1,2]


Constraints:

The number of nodes in the list is in the range [0, 200].
-100 <= Node.val <= 100
-200 <= x <= 200
"""

from data_structures.lists import SingleLinkNode as ListNode


def partition_inplace(head: ListNode | None, x: int) -> ListNode | None:
    left_head, right_head = ListNode(None), ListNode(None)
    left_tail, right_tail = left_head, right_head

    curr = head
    while curr:
        if curr.val < x:
            left_tail.next = curr
            left_tail = left_tail.next
        else:
            right_tail.next = curr
            right_tail = right_tail.next
        curr = curr.next

    head = left_head.next  # side-effect: resulting list and head will be the same
    left_tail.next = right_head.next
    right_tail.next = None  # terminate the list, break the cycle
    return head


if __name__ == "__main__":
    head = ListNode.from_array([1, 4, 3, 2, 5, 2])
    k = 3
    print(partition_inplace(head, k).serialize())  # [1,2,2,4,3,5]
