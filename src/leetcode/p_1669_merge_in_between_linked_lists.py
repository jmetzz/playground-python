"""
https://leetcode.com/problems/merge-in-between-linked-lists/description

1669. Merge In Between Linked Lists
Medium
Topics: #linked-lists

You are given two linked lists: list1 and list2 of sizes n and m respectively.

Remove list1's nodes from the ath node to the bth node, and put list2 in their place.

The blue edges and nodes in the following figure indicate the result:


Build the result list and return its head.



Example 1:


Input: list1 = [10,1,13,6,9,5], a = 3, b = 4, list2 = [1000000,1000001,1000002]
Output: [10,1,13,1000000,1000001,1000002,5]
Explanation: We remove the nodes 3 and 4 and put the entire list2 in their place.
The blue edges and nodes in the above figure indicate the result.

Example 2:
Input: list1 = [0,1,2,3,4,5,6], a = 2, b = 5, list2 = [1000000,1000001,1000002,1000003,1000004]
Output: [0,1,1000000,1000001,1000002,1000003,1000004,6]
Explanation: The blue edges and nodes in the above figure indicate the result.


Constraints:

3 <= list1.length <= 104
1 <= a <= b < list1.length - 1
1 <= list2.length <= 104

"""


from data_structures.lists import SingleLinkNode as ListNode


def merge_in_between(list1: ListNode, a: int, b: int, list2: ListNode) -> ListNode:
    if not list1:
        return list2
    dummy_head = ListNode(next_node=list1)
    left = dummy_head
    steps = 0
    while left and steps < a:
        # advance left pointer until just before
        # the target node 'a'
        left = left.next
        steps += 1

    right = left
    while right and steps <= b:
        # advance left pointer until reaching
        # the target node 'b'
        right = right.next
        steps += 1

    # connect the list2 to the right of the left pointer
    left.next = list2

    # move the left pointer until the end of list2
    while left.next:
        left = left.next

    # connect the end of list2 to
    # the right elements from original list1, if they exist
    left.next = right.next if right else None

    # return the new head
    return dummy_head.next
