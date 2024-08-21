"""https://leetcode.com/problems/remove-zero-sum-consecutive-nodes-from-linked-list/
1171. Remove Zero Sum Consecutive Nodes from Linked List
Medium

Given the head of a linked list, we repeatedly delete consecutive
sequences of nodes that sum to 0 until there are no such sequences.

After doing so, return the head of the final linked list.
You may return any such answer.

(Note that in the examples below, all sequences are serializations of ListNode objects.)

Example 1:
Input: head = [1,2,-3,3,1]
Output: [3,1]
Note: The answer [1,2,1] would also be accepted.

Example 2:
Input: head = [1,2,3,-3,4]
Output: [1,2,4]

Example 3:
Input: head = [1,2,3,-3,-2]
Output: [1]


Constraints:
The given linked list will contain between 1 and 1000 nodes.
Each node in the linked list has -1000 <= node.val <= 1000.
"""

from typing import Optional

from data_structures.lists import SingleLinkNode as ListNode


def remove_zero_sum_sublists(head: Optional[ListNode]) -> Optional[ListNode]:
    """Removes all contiguous sublists that sum up to zero from a given linked list.

    This function iterates through the linked list, maintaining a running prefix sum.
    It utilizes a dictionary to track previously encountered sums and their corresponding nodes.
    When a repeated sum is encountered, it indicates the end of a zero-sum sublist,
    which is then removed. This process is repeated until the entire list is processed.

    Args:
    ----
        head (Optional[ListNode]): The head of the input linked list.

    Returns:
    -------
        Optional[ListNode]: The head of the modified linked list after removing
                            zero-sum sublists.

    Example:
    -------
        Given the linked list 3 -> 4 -> -7 -> 5 -> -6 -> 6, it contains the sublists
        3 -> 4 -> -7 and 5 -> -6 -> 6, which sum up to zero. After removing these
        sublists, the modified list will be empty.

        Given the linked list 1 -> 2 -> -3 -> 3 -> 1, it contains the sublist 2 -> -3,
        which sums up to zero. Removing this sublist, the modified list will be
        1 -> 3 -> 1.

    Note:
    ----
        The input linked list may contain multiple, overlapping zero-sum sublists.
        This function ensures all such sublists are removed, and the remaining nodes
        are returned in their original order.

    """
    dummy = ListNode(next_node=head)
    sums_dict = {0: dummy}
    prefix_sum = 0

    current = head
    while current:
        prefix_sum += current.val
        if prefix_sum in sums_dict:
            to_delete = sums_dict[prefix_sum].next
            temp_sum = prefix_sum
            while to_delete != current:
                temp_sum += to_delete.val  # add the val back since we are removing the node
                del sums_dict[temp_sum]
                to_delete = to_delete.next
            sums_dict[prefix_sum].next = current.next
        else:
            sums_dict[prefix_sum] = current
        current = current.next

    return dummy.next
