"""https://leetcode.com/problems/find-the-pivot-integer/description
2485. Find the Pivot Integer
Easy

Given a positive integer n, find the pivot integer x such that:

The sum of all elements between 1 and x inclusively equals the
sum of all elements between x and n inclusively.
Return the pivot integer x. If no such integer exists, return -1.
It is guaranteed that there will be at most one pivot index for the given input.



Example 1:
Input: n = 8
Output: 6
Explanation: 6 is the pivot integer since: 1 + 2 + 3 + 4 + 5 + 6 = 6 + 7 + 8 = 21.

Example 2:
Input: n = 1
Output: 1
Explanation: 1 is the pivot integer since: 1 = 1.

Example 3:
Input: n = 4
Output: -1
Explanation: It can be proved that no such integer exist.


Constraints:

1 <= n <= 1000
"""

import timeit
from math import sqrt


def pivot_brute_force(n: int) -> int:
    """Finds a pivot element in the range [1, n]

    such that the sum of elements up to (and including) the pivot
    is equal to the sum of elements from the pivot to n.

    This function uses a brute-force approach to find the pivot element.
    It iterates through each possible pivot value in the range [1, n] and
    calculates the sum of elements on both sides of the pivot.
    If a pivot value is found where the sum of elements on the left side
    equals the sum of elements on the right side, that pivot value is returned.

    Args:
    ----
        n (int): The upper bound of the range within which to search for the pivot element.
        n should be a positive integer.

    Returns:
    -------
        int: The pivot element for which the sum of elements on its left equals the
        sum of elements on its right. If no such element exists, returns -1.

    Examples:
    --------
        >>> pivot_brute_force(5)
        -1

        >>> pivot_brute_force(3)
        2

    Note:
    ----
        This brute-force approach can be inefficient for large values of n
        due to the repetitive computation of sums on each iteration.
        It is primarily for demonstration or when n is known to be small.

    """
    # Iterate through possible pivot values
    for i in range(1, n + 1):
        # Calculate the sum of elements on the left side of the pivot
        sum_left = sum(range(1, i + 1))
        # Calculate the sum of elements on the right side of the pivot
        sum_right = sum(range(i, n + 1))

        # Check if the sums on both sides are equal
        if sum_left == sum_right:
            return i  # Return the pivot value if found

    return -1  # Return -1 if no pivot is found


def pivot_prefix_sum(n: int) -> int:
    """Finds a pivot element in the range [1, n]

    such that the sum of elements on the left is equal to the sum of elements on the right.
    This function computes the total sum of integers from 1 to n using the formula
    for the sum of an arithmetic sequence.
    It then iterates over each element `e` in the range [1, n],
    calculating the cumulative sum (`left_sum`) as it goes.
    For each element, it checks if the sum of elements to the left of `e`
    is equal to the sum of elements to the right of `e`.
    If such an element is found, it is returned as the pivot element; otherwise,
    the function returns -1, indicating no such pivot exists.

    Args:
    ----
        n (int): The upper bound of the range within which to search for the pivot element.
        n should be a positive integer.

    Returns:
    -------
        int: The pivot element for which the sum of elements on its left is equal to
        the sum of elements on its right. If no such element exists, returns -1.

    Examples:
    --------
        >>> pivot_prefix_sum(5)
        -1

        >>> pivot_prefix_sum(4)
        -1

    """
    # Calculate the total sum of the sequence using the sum of arithmetic series formula
    total_sum = (n * (n + 1)) // 2
    # the prefix sum as we iterate over the elements
    left_sum = 0
    for e in range(1, n + 1):
        if left_sum == total_sum - (left_sum + e):
            return e
        left_sum += e
    return -1


def pivot_mathematical_solution(n: int) -> int:
    """Finds an integer x within the range [1, n]

    such that the sum of integers from 1 to x equals the sum of integers from x to n.
    This function calculates x by solving the quadratic equation derived from
    the sum of arithmetic sequences.

    The sum of integers from 1 to x can be represented as (1 + x) * x / 2.
    The problem statement implies finding x where this sum equals the
    sum of the remaining sequence to n, leading to the equation:
        2 * x^2 = n^2 + n

    and solving for x gives us

    1 + 2 + ... + x  = x + ... + n
    (1 + x) * x // 2 = (x + n) * (n - x + 1) // 2
             x + x^2 = nx - x^2 + x + n^2 - nx + n
             2 * x^2 = n^2 + n
                   x = sqrt((n^2 + n) // 2)

    For improved computational time:
        x = sqrt((n * (n + 1)) // 2)

    If x is an interger number (in this case with the float point portion equals to zero),
    than x is the pivot point. Otherwise, there is no pivot point for n.

    Args:
    ----
        n (int): The upper bound of the range to search for the integer x.
        n should be a non-negative integer.

    Returns:
    -------
        int: The integer x that satisfies the sum property.
        If no such integer x exists within the range [1, n]
        or the solution is not an integer, the function returns -1.

    Example:
    -------
        >>> pivot_mathematical_solution(8)
        6

        >>> pivot_mathematical_solution(5)
        -1

    Note:
    ----
        This function uses a mathematical approach to identify the pivot point,
        relying on algebraic manipulation and solving a quadratic equation to
        find the potential value of x.
        The final result is checked to ensure it is a valid integer within
        the specified range.
        Note that rounding issues might happen.

    """
    """

    """
    x = sqrt(n * (n + 1) / 2)
    int_x = int(x)
    return int_x if int_x == x else -1


def pivot_bin_search(n: int) -> int:
    """Finds a pivot number within the range [1, n]

    whose square equals the sum of the sequence 1 to n.

    This function performs a binary search to find the pivot point,
    if it exists, for which the square of the pivot is equal to the
    sum of all integers from 1 to n.
    The sum of integers from 1 to n is calculated using the formula:
        n * (n + 1) / 2

    If such a pivot does not exist, the function returns -1.

    Args:
    ----
        n (int): The upper bound of the range to search for the pivot.
                 n should be a positive integer.

    Returns:
    -------
        int: The pivot number whose square equals the sum of integers from 1 to n.
        Returns -1 if no such pivot exists.

    Example:
    -------
        >>> pivot_bin_search(8)
        6
        Explanation: For n=8, the sum of integers from 1 to 8 is 36.
        So, we are looking for a number x in the range [1, 8] for which x^2 = 36.
        The number that satisfies this equation is x = 6, because 6^2 = 36.

        >>> pivot_bin_search(5)
        -1
        Explanation: For n=5, the sum of integers from 1 to 5 is 15.
        There is no integer within the range [1, 5] whose square equals 15,
        so the function returns -1.

        >>> pivot_bin_search(16)
        -1
        Explanation: For n=16, the sum of integers from 1 to 16 is 136.
        There is no integer within the range [1, 16] whose square equals 136,\
        so the function returns -1.

    """
    # Initialize left and right pointers for binary search
    left, right = 1, n

    # Calculate the total sum of the sequence 1 to n
    total_sum = n * (n + 1) // 2

    # Perform binary search to find the pivot point
    while left < right:
        # Calculate the mid-point of the current search range
        mid = (left + right) // 2

        # Check if the square of mid is less than the total sum
        if mid * mid < total_sum:
            # If so, adjust the left bound to narrow the search range
            left = mid + 1
        else:
            # Otherwise, adjust the right bound
            right = mid

    # After exiting the loop, 'left' is the candidate for the pivot.
    # Verify it satisfies the condition. If affirmative, the pivot, otherwise return -1.
    return left if left * left == total_sum else -1


if __name__ == "__main__":
    pivot_functions = [pivot_brute_force, pivot_prefix_sum, pivot_bin_search, pivot_mathematical_solution]
    input_n = 5
    for func in pivot_functions:
        print(func(input_n))

    num_runs = 100
    print(">>> Performance evaluation: ")
    for func in pivot_functions:
        time_taken = timeit.timeit(
            stmt=f"{func.__name__}(n)",  # Use a slice of elements to avoid modifying the original list
            setup=f"from __main__ import {func.__name__}; n = {input_n}",
            globals=globals(),  # Provide the global namespace so timeit can access the functions and variables
            number=num_runs,
        )
        print(f"'{func.__name__}' took " f"{time_taken:.5f} seconds")
