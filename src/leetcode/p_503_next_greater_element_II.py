"""https://leetcode.com/problems/next-greater-element-ii/description/
503. Next Greater Element II

Medium

Given a circular integer array nums (i.e., the next element of nums[nums.length - 1] is nums[0]),
return the next greater number for every element in nums.

The next greater number of a number x is the first greater number to its traversing-order
next in the array, which means you could search circularly to find its next greater number.
If it doesn't exist, return -1 for this number.



Example 1:
Input: nums = [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2;
The number 2 can't find next greater number.
The second 1's next greater number needs to search circularly, which is also 2.

Example 2:
Input: nums = [1,2,3,4,3]
Output: [2,3,4,-1,4]

Constraints:

1 <= nums.length <= 104
-109 <= nums[i] <= 109
"""

from typing import List


def next_greater_elements_1(nums: List[int]) -> List[int]:
    n = len(nums)
    answer = [-1] * n

    for i in range(n):
        j = (i + 1) % n
        while nums[i] >= nums[j] and j != i:
            j = (j + 1) % n
        if i != j:
            answer[i] = nums[j]
    return answer


def next_greater_elements_2(nums: List[int]) -> List[int]:
    """Two loops and a decreasing monotonic stack to solve this problem.

    In order to properly handle the circular nature of the array we need to use
    two iterations over the input list, and make use of a decresing monotonic stack
    to keep track of the elements.

    The circular nature of the array means that after reaching the end of the array,
    the search for the next greater element continues from the beginning of the array.
    A single pass will potentially leave some elements without their next greater
    value identified, especially if it's located before them in the array and after them
    in the circular traversal order.

    Therefore, a second loop gives these elements a chance to find their respective
    next greater element in the circular array.

    First Loop:
    * The decreasing monotonic stack stores indices of the elements, not the elements themselves.
    * Any new element that is greater than the element corresponding to the index at the top
    of the stack is the next greater element for the one at the top of the stack.
    * As we iterate through the array, you compare the current element with the element
    at the index at the top of the stack. If the current element is greater, it's the
    next greater element for those indices we have popped from the stack.
    Example:  nums = [2, 1, 2, 4, 3]
        Start with an empty stack and an answer array filled with -1s,
        indicating that no next greater element has been found yet: answer = [-1, -1, -1, -1, -1].

        Iterate over each element in nums:
        2: Stack is empty, push its index 0 to the stack.
        1: 1 < 2, so push its index 1.
        2: Pop 1 because 2 > nums[1], set answer[1] = 2. Push index 2.
        4: Pop 2 and 0 because 4 > nums[2] and 4 > nums[0], set answer[2] = 4 and answer[0] = 4. Push index 3.
        3: Doesn't pop anything because 3 < 4, push index 4.
        After the first loop, the stack is [3, 4], and answer is [4, 2, 4, -1, -1].
        Notice that the elements at indices 3 and 4 haven't found their next greater element yet.

    Second Loop:
    * The purpose of the second loop is to ensure that elements that didn't find their
    next greater element in the first pass get a chance to find it.
    * After the first loop, the stack contains indices of elements in decreasing order.
    * The idea is to continue using the current state of the stack and the elements in
    nums to fill in the remaining -1s in answer.

    Example:
    -------
        After the first loop, the partial answer and the stack are:
        nums = [2, 1, 2, 4, 3]; answer = [4, 2, 4, -1, -1]; stack = [3, 4]

        The second loop revisits elements from the start (because of the circular condition),
        allowing us to update:
        * For index 3 (nums[3] = 4), there's no greater element, so it remains -1.
        * For index 4 (nums[4] = 3), the next greater element value is 4
        (it wraps around and finds the 4 at index 3, ie, nums[3] = 4), updating answer[4] = 4.

        Final Answer is:
        answer = [4, 2, 4, -1, 4]

    """
    stack = []  # keep elements already seen
    answer = [-1] * len(nums)

    for idx, n in enumerate(nums):
        while stack and n > nums[stack[-1]]:
            # n is the first element greater than the top of the stack
            answer[stack.pop()] = n
        stack.append(idx)

    for n in nums:
        while stack and n > nums[stack[-1]]:
            answer[stack.pop()] = n

    return answer
