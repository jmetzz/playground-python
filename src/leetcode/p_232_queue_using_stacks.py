"""232. Implement Queue using Stacks
Easy

Implement a first in first out (FIFO) queue using only two stacks.
The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:
- void push(int x) Pushes element x to the back of the queue.
- int pop() Removes the element from the front of the queue and returns it.
- int peek() Returns the element at the front of the queue.
- boolean empty() Returns true if the queue is empty, false otherwise.

Notes:
-----
You must use only standard operations of a stack, which means only push to top,
peek/pop from top, size, and is empty operations are valid.
Depending on your language, the stack may not be supported natively.
You may simulate a stack using a list or deque (double-ended queue)
as long as you use only a stack's standard operations.


Example 1:
Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation:
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false

"""

from typing import Optional


class QueueUsingStacks:
    """Push - O(1) per operation,
    Pop - Amortized O(1) per operation.

    Push:
        The newly arrived element is always added on top of stack s1 (self._stack) and
         the first element is kept as front queue element
    """

    def __init__(self):
        self._stack = []
        self._aux = []

    def push(self, x: int) -> None:
        """Time complexity : O(1). Appending an element to a stack is an O(1) operation."""
        self._stack.append(x)

    def pop(self) -> Optional[int]:
        return self._peek(keep=False)

    def peek(self) -> Optional[int]:
        return self._peek(True)

    def _peek(self, keep: bool) -> Optional[int]:
        """Pop and Peek operation only differs by the flag keep.
        If keep is True, then perform peek operation
        If keep is False, perform pop operation.

        We have to remove element in front of the queue.
        This is the first inserted element in the stack s1 (self._stack) and
        it is positioned at the bottom of the stack because of
        stack's LIFO (last in - first out) policy.

        To remove the bottom element from s1, we have to pop
        all elements from s1 (self._stack) and to push them on to an
        additional stack s2 (self._aux), which helps us to store
        the elements of self._stack in reversed order.

        This way the bottom element of self._stack  will be positioned on
        top of self._aux and we can simply pop it from there.
        Once self._stack is empty, the algorithm transfer data
        from self._aux to self._stack again to complete the operation.
        """
        if len(self._stack) == 0:
            return None
        while self._stack:  # while not empty
            # move all elements onto the auxiliary stack
            self._aux.append(self._stack.pop())
        result = self._aux.pop()
        if keep:
            self._stack.append(result)
        # transfer back the elements on the main stack
        while self._aux:
            self._stack.append(self._aux.pop())
        return result

    def empty(self) -> bool:
        return len(self._stack) == 0
