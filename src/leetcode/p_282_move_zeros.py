"""283. Move Zeroes
Easy

Given an integer array nums, move all 0's to the end of it
while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.


Example 1:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]

Example 2:
Input: nums = [0]
Output: [0]

Constraints:
1 <= nums.length <= 104
-231 <= nums[i] <= 231 - 1

Follow up: Could you minimize the total number of operations done?
"""

import random
import timeit


def move_zeroes_loops(nums: list[int]) -> None:
    """Moves all 0s to the end of the array while maintaining
    the relative order of the non-zero elements using two loops.

    This function iterates over the array, placing all non-zero elements at
    the beginning. It keeps track of the position where the next non-zero
    element should be placed. After processing all elements,
    it fills the remainder of the array with 0s,
    effectively moving all zeros to the end.

    The first loop places non-zero elements at the beginning of the array.
    The second loop fills the rest of the array with zeros.

    Args:
    ----
        nums (List[int]): The input array containing zero and non-zero integers.

    Modifies:
        The input array `nums` is modified in-place, with all zeros moved to the end
        while maintaining the order of non-zero elements.

    Examples:
    --------
        >>> nums = [0, 1, 0, 3, 12]
        >>> move_zeroes_loops(nums)
        >>> nums
        [1, 3, 12, 0, 0]

    """
    next_spot = 0
    for num in nums:
        if num != 0:
            nums[next_spot] = num
            next_spot += 1
    while next_spot < len(nums):
        nums[next_spot] = 0
        next_spot += 1


def move_zeroes_fast_slow(nums: list[int]) -> None:
    """Moves all 0s to the end of the array while maintaining
    the relative order of the non-zero elements

    This approach uses two pointers (`slow` and `fast`) to efficiently
    swap zero elements found by the `slow` pointer with non-zero elements
    found by the `fast` pointer.

    The `slow` pointer is only incremented when a non-zero element
    is placed in its correct position, ensuring that all non-zero
    elements are moved to the front of the array without breaking
    their original order.

    Args:
    ----
        nums (List[int]): The input array containing zero and non-zero integers.

    Modifies:
        The input array `nums` is modified in-place, with all zeros moved to the end
        while preserving the order of non-zero elements.

    Examples:
    --------
        >>> nums = [0, 1, 0, 3, 12]
        >>> move_zeroes_fast_slow(nums)
        >>> nums
        [1, 3, 12, 0, 0]

    Note:
    ----
        This method is particularly efficient because it minimizes the number of
        operations and swaps only when necessary, making it an optimal choice
        for large arrays or arrays with few zeros.

    """
    n = len(nums)
    slow = 0
    for fast in range(n):
        if nums[slow] == 0 and nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]
        if nums[slow] != 0:
            slow += 1


if __name__ == "__main__":
    num_items = 100000
    val_range = (-(2**31), 2**31 - 1)
    nums = [random.randint(*val_range) for _ in range(num_items)]

    loops_time = timeit.timeit(
        "move_zeroes_loops(nums[:])",
        globals=globals(),
        number=100,
    )
    fast_slow_time = timeit.timeit("move_zeroes_fast_slow(nums[:])", globals=globals(), number=100)

    print(f"move_zeroes_loops: {loops_time} seconds")
    print(f"move_zeroes_fast_slow: {fast_slow_time} seconds")
