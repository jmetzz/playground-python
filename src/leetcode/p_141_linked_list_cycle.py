"""
https://leetcode.com/problems/linked-list-cycle/description/

141. Linked List Cycle
Easy

Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by
continuously following the next pointer.
Internally, pos is used to denote the index of the node that tail's next pointer is connected to.
Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.

Example 1:
Input: head = [3,2,0,-4], pos = 1
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

Example 2:
Input: head = [1,2], pos = 0
Output: true
Explanation: There is a cycle in the linked list, where the tail connects to the 0th node.

Example 3:
Input: head = [1], pos = -1
Output: false
Explanation: There is no cycle in the linked list.


Constraints:
The number of the nodes in the list is in the range [0, 104].
-105 <= Node.val <= 105
pos is -1 or a valid index in the linked-list.

Follow up: Can you solve it using O(1) (i.e. constant) memory?
"""
from typing import Optional

from data_structures import SingleLinkNode


def has_cycle(head: Optional[SingleLinkNode]) -> bool:
    """Implements the Floyd’s Cycle-Finding Algorithm

    This method is also known as the "fast-slow approach" or  "tortoise and the hare algorithm".

    Approach: use two pointers to traverse the list while checking if they are ever pointing to the same node.
        Use a fast and a slow pointer.
        The fast pointer move 2 nodes each cycle, while the slow pointer moves 1 node each cycle.
        If these pointers meet at the same node then there is a loop. Otherwise, there is no loop.

    Args:
        head: the first node of the list

    Returns:
        True if there is a cycle in the list, False otherwise
    """

    slow, fast = head, head

    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
        if fast == slow:
            return True

    return False
