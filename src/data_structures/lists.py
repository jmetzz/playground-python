from typing import Any, List, Optional, Self


class SingleLinkNode:
    def __init__(self, val: Any = None, next_node: Self = None):
        self.val = val
        self.next = next_node

    @classmethod
    def from_array(cls, arr: List[Any]) -> Self:
        """
        Recursively creates a singly-linked list from an array of integers.

        Args:
            arr (List[int]): The array of integers to create the list from.
            index (int): The current index in the array being processed.

        Returns:
            ListNode: The head of the singly-linked list.
        """
        if not arr:
            return None

        root = SingleLinkNode(arr[0])
        root.next = cls.from_array(arr[1:])
        return root

    @staticmethod
    def _create_iterative(arr):
        """
        Creates a singly-linked list from an array of integers.

        Args:
            arr (List[int]): The array of integers to create the list from.

        Returns:
            ListNode: The head of the singly-linked list.
        """
        head = None
        current = None
        for val in arr:
            if head is None:
                head = SingleLinkNode(val)
                current = head
            else:
                current.next = SingleLinkNode(val)
                current = current.next
        return head

    @classmethod
    def deserialize(cls, arr: List[Any], size_threshold: int = 500) -> Optional[Self]:
        if not arr:
            return None
        if len(arr) > size_threshold:
            return cls._create_iterative(arr)
        return cls.from_array(arr)

    def serialize(self) -> str:
        return str(self.as_array())

    def as_array(self) -> List[Any]:
        arr = []
        visitor = self
        while visitor:
            arr.append(visitor.val)
            visitor = visitor.next
        return arr

    def __str__(self) -> str:
        values = []
        node = self
        while node:
            values.append(node.val)
            node = node.next
        return " -> ".join(values)

    def equal(self, other: Self) -> bool:
        head1 = self
        head2 = other
        while head1 and head2:
            if head1.val != head2.val:
                return False
            head1, head2 = head1.next, head2.next
        return head1 is None and head2 is None
