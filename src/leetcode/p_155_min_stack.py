"""
https://leetcode.com/problems/min-stack/description

155. Min Stack

My solution
https://leetcode.com/problems/min-stack/solutions/5714931/simple-and-intuitive-python-implementation-o-1-operations/

Medium
Topics
Companies
Hint
Design a stack that supports push, pop, top, and retrieving
the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.



Example 1:

Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]

Output
[null,null,null,null,-3,null,0,-2]

Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2


Constraints:

-2^31 <= val <= 2^31 - 1
Methods pop, top and getMin operations will always be called on non-empty stacks.
At most 3 * 104 calls will be made to push, pop, top, and getMin.
"""

from collections import namedtuple


class MinStack:
    """
    Stack data structure that supports retrieving the minimum element
    in constant time.

    Attributes:
        _stack (list): Internal list used to store the stack items,
                       where each item is a tuple containing
                       the value and the current minimum value from
                       this position down the stack.
    """

    Item = namedtuple("Item", ["val", "min"])

    def __init__(self):
        """Initializes an empty MinStack."""
        self._stack = []

    def push(self, val: int) -> None:
        """
        Pushes a new element onto the stack.

        Args:
            val (int): The value to be pushed onto the stack.
        """
        _min = min(val, self._stack[-1].min) if self._stack else val
        self._stack.append(MinStack.Item(val, _min))

    def pop(self) -> None:
        """
        Removes the element on the top of the stack.

        """
        if self._stack:
            self._stack.pop()

    def top(self) -> int | None:
        """
        Gets the top element of the stack.

        Returns:
            int | None: The value of the top element
            if the stack is not empty, otherwise None.
        """
        return self._stack[-1].val if self._stack else None

    def min(self) -> int | None:
        """
        Retrieves the minimum element in the stack.

        Returns:
            int | None: The minimum value in the stack
            if the stack is not empty, otherwise None.
        """
        return self._stack[-1].min if self._stack else None

    def is_empty(
        self,
    ) -> bool:
        """
        Checks whether the stack is empty.

        Returns:
            bool: True if the stack is empty, False otherwise.
        """
        return len(self._stack) == 0


if __name__ == "__main__":
    stack = MinStack()
    stack.push(-2)
    stack.push(0)
    stack.push(-3)
    print(stack.min())
    print(stack.pop())
    print(stack.top())
    print(stack.min())
