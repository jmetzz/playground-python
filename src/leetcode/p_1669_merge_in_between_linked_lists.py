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
