from data_structures.lists import SingleLinkNode as ListNode


def delete_middle(head: ListNode | None) -> ListNode | None:
    """Fast and slow pointers + dummy node

    Use two pointers and dummy node technique.

    dummy1 ---> [1 ---> 2 ---> 3 ---> 4 ---> 5]
    ^slow
                 ^fast

    Move fast until reaching the end of the list.
    At which point slow.next points to the target node.

    Then we can remove the target node using slow.next.


    dummy1 --> 1 ---> 3 ---> 4 ---> |7| ---> 1 ---> 2 ---> 5  .
                                     ^                        ^
                                     slow                    fast

    dummy1 --> 1 ---> 3 ---> 4 ---> |7| ---> 1 ---> 2  .
                                     ^                 ^
                                    slow              fast

    Args:
    ----
        head: the node at the first position

    Returns:
    -------
        the head of the list

    """
    if not head or not head.next:
        return None

    dummy_2 = ListNode()
    dummy_2.next = head

    slow, fast = dummy_2, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next

    # slow points to the node just before the target node for removal
    target_node = slow.next
    slow.next = target_node.next
    # clean up the connection from the removed node to the list
    target_node.next = None
    return dummy_2.next


if __name__ == "__main__":
    inputs = [
        [],
        [1],
        [2, 3],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 3, 4, 7, 1, 2, 6],
    ]

    for arr in inputs:
        h = delete_middle(ListNode.from_array(arr))
        print(h.serialize() if h else [])
        print("---")
