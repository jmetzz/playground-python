"""https://leetcode.com/problems/maximum-twin-sum-of-a-linked-list/description

2130. Maximum Twin Sum of a Linked List
Medium

In a linked list of size n, where n is even, the ith node (0-indexed) of
the linked list is known as the twin of the (n-1-i)th node, if 0 <= i <= (n / 2) - 1.

For example, if n = 4, then node 0 is the twin of node 3, and node 1 is
the twin of node 2. These are the only nodes with twins for n = 4.
The twin sum is defined as the sum of a node and its twin.

Given the head of a linked list with even length, return the maximum
twin sum of the linked list.



Example 1:
Input: head = [5,4,2,1]
Output: 6
Explanation:
Nodes 0 and 1 are the twins of nodes 3 and 2, respectively. All have twin sum = 6.
There are no other nodes with twins in the linked list.
Thus, the maximum twin sum of the linked list is 6.

Example 2:
Input: head = [4,2,2,3]
Output: 7
Explanation:
The nodes with twins present in this linked list are:
- Node 0 is the twin of node 3 having a twin sum of 4 + 3 = 7.
- Node 1 is the twin of node 2 having a twin sum of 2 + 2 = 4.
Thus, the maximum twin sum of the linked list is max(7, 4) = 7.

Example 3:
Input: head = [1,100000]
Output: 100001
Explanation:
There is only one node with a twin in the linked list having twin sum of 1 + 100000 = 100001.


Constraints:

The number of nodes in the list is an even integer in the range [2, 105].
1 <= Node.val <= 105
"""

from typing import Optional

from data_structures.lists import SingleLinkNode as ListNode


def pair_sum_fast_slow(head: Optional[ListNode]) -> int:
    """Calculates the maximum twin sum in a linked list by using a stack
    to store values of the first half while traversing the list with two pointers.

    The first half of the values are pushed onto a stack as the 'slow' pointer advances.
    Once the end of the list is reached, elements are popped from the stack and added
    to the corresponding 'slow' pointer values, calculating the twin sums.

    Args:
    ----
        head (Optional[ListNode]): The head of the linked list.

    Returns:
    -------
        int: The maximum twin sum in the linked list.

    """
    stack = []
    fast, slow = head, head

    while fast:
        stack.append(slow.val)
        slow = slow.next
        fast = fast.next.next

    max_twin_sum = float("-inf")
    while slow:
        twin = stack.pop()
        max_twin_sum = max(max_twin_sum, twin + slow.val)
        slow = slow.next
    return max_twin_sum


def pair_sum_reverse_first_half(head: Optional[ListNode]) -> int:
    """Calculates the maximum twin sum in a linked list by first reversing the first half
    of the list, then comparing each element of the reversed first half with its
    corresponding element in the second half.

    The list is split in half and the first half is reversed. The twin sum is calculated
    by comparing elements from the reversed first half and the non-reversed second half.

    Args:
    ----
        head (Optional[ListNode]): The head of the linked list.

    Returns:
    -------
        int: The maximum twin sum in the linked list.

    """
    slow, fast = head, head
    prev = None
    max_twin_sum = 0

    while fast and fast.next:
        fast = fast.next.next
        next_node = slow.next
        slow.next = prev
        prev = slow
        slow = next_node

    while slow:
        max_twin_sum = max(max_twin_sum, prev.val + slow.val)
        prev = prev.next
        slow = slow.next
    return max_twin_sum


def pair_sum_reverse_second_half(head: Optional[ListNode]) -> int:
    """Calculates the maximum twin sum in a linked list by reversing the second half
    of the list and then comparing each element of the first half with its corresponding
    element in the reversed second half.

    After finding the midpoint of the list, the second half of the list is reversed.
    The twin sum is calculated by comparing elements from the non-reversed first half
    and the reversed second half.

    Args:
    ----
        head (Optional[ListNode]): The head of the linked list.

    Returns:
    -------
        int: The maximum twin sum in the linked list.

    """
    slow, fast = head, head
    max_twin_sum = float("-inf")

    # Find the midpoint of the list
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    # Reverse second half of the list
    current, prev = slow, None
    while current:
        next_node = current.next  # Store the next node
        current.next = prev  # Reverse the current node's pointer
        prev = current  # Move prev one step forward
        current = next_node  # Move current one step forward

    # Calculate twin sums using the original first half (head)
    # and the reversed second half (prev)

    while prev:
        twin_sum = prev.val + head.val
        max_twin_sum = max(max_twin_sum, twin_sum)
        prev = prev.next
        head = head.next

    return max_twin_sum
