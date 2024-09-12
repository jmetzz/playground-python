"""
https://leetcode.com/problems/rotate-list/description
61. Rotate List

Medium
Given the head of a linked list, rotate the list to the right by k places.

Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]

Example 2:
Input: head = [0,1,2], k = 4
Output: [2,0,1]

Constraints:

The number of nodes in the list is in the range [0, 500].
-100 <= Node.val <= 100
0 <= k <= 2 * 10^9
"""

from data_structures.lists import SingleLinkNode as ListNode


def rotate_right(head: ListNode | None, k: int) -> ListNode | None:
    if not head:
        return None

    list_size, tail = 1, head
    while tail.next:
        tail = tail.next
        list_size += 1

    k %= list_size  # to handle k > list_size
    if k == 0:
        # multiple of the list_size
        return head

    breaking_position = list_size - k - 1
    curr = head
    for _ in range(breaking_position):
        curr = curr.next

    tail.next = head
    head = curr.next
    curr.next = None

    return head


if __name__ == "__main__":
    head = ListNode.from_array([1, 2, 3, 4, 5])
    k = 7
    print(rotate_right(head, k).serialize())
