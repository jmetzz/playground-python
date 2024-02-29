"""
https://leetcode.com/problems/middle-of-the-linked-list/description/
876. Middle of the Linked List
Easy

Given the head of a singly linked list, return the middle node of the linked list.

If there are two middle nodes, return the second middle node.

Example 1:
Input: head = [1,2,3,4,5]
Output: [3,4,5]
Explanation: The middle node of the list is node 3.

Example 2:
Input: head = [1,2,3,4,5,6]
Output: [4,5,6]
Explanation: Since the list has two middle nodes with values 3 and 4, we return the second one.


Constraints:
The number of nodes in the list is in the range [1, 100].
1 <= Node.val <= 100

"""
from typing import Optional

from data_structures import SingleLinkListNode


def middle_node(head: Optional[SingleLinkListNode]) -> Optional[SingleLinkListNode]:
    stack = []
    pointer = head
    while pointer:
        stack.append(pointer)
        pointer = pointer.adjacent
    mid = len(stack) // 2
    return stack[mid]


if __name__ == "__main__":
    print(middle_node(SingleLinkListNode.from_array([1, 2, 3, 4, 5])).val)
    print(middle_node(SingleLinkListNode.from_array([1, 2, 3, 4, 5, 6])).val)
