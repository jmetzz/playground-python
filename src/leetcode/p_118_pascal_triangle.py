"""https://leetcode.com/problems/pascals-triangle/description/
118. Pascal's Triangle
Easy

Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


Example 1:
Input: numRows = 5
Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

Example 2:
Input: numRows = 1
Output: [[1]]


Constraints:

1 <= numRows <= 30
"""

from typing import List


def generate_iterative(num_rows: int) -> List[List[int]]:
    """Generates the first `num_rows` of Pascal's Triangle using an iterative approach.

    This function constructs each row of Pascal's Triangle one by one, starting from
    the first row. Each row is built based on the previous row's values, following
    Pascal's rule that each number is the sum of the two numbers directly above it.

    Args:
    ----
        num_rows (int): The number of rows of Pascal's Triangle to generate.

    Returns:
    -------
        List[List[int]]: A list of lists, where each sublist represents a row of Pascal's Triangle.

    Examples:
    --------
        >>> generate_iterative(5)
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

    """
    if num_rows == 0:
        return []
    triangle = [[1]]  # row zero
    for row in range(1, num_rows):
        row_elements = [1] * (row + 1)
        for i in range(1, len(row_elements) - 1):
            row_elements[i] = triangle[row - 1][i - 1] + triangle[row - 1][i]
        triangle.append(row_elements)
    return triangle


def generate_iterative_padding(num_rows: int) -> List[List[int]]:
    """Generate the first num_rows of Pascal's triangle using an iterative approach.

    This function builds each row of Pascal's triangle by leveraging a temporary list
    that pads the previous row with zeros at both ends. This simplification allows for
    direct calculation of each new row's elements as the sum of pairs of elements from
    this padded list. The approach efficiently constructs Pascal's triangle row by row.

    Args:
    ----
        num_rows (int): The number of rows of Pascal's triangle to generate.

    Returns:
    -------
        List[List[int]]: A list of lists, where each inner list represents a row of
                         Pascal's triangle up to the specified number of rows.

    Examples:
    --------
        >>> generate_iterative_2(1)
        [[1]]

        >>> generate_iterative_2(5)
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

    """
    if num_rows == 0:
        return []
    triangle = [[1]]  # row zero
    for i in range(1, num_rows):
        temp = [0] + triangle[i - 1] + [0]  # Pad the previous row with zeros.
        row_elements = []
        for j in range(i + 1):
            row_elements.append(temp[j] + temp[j + 1])  # Calculate new row elements.
        triangle.append(row_elements)
    return triangle


def generate_recursive(num_rows: int, triangle=None) -> List[List[int]]:
    """Generates the first `num_rows` of Pascal's Triangle using a recursive approach.

    This function employs recursion by building each row of Pascal's Triangle based on the previous row,
    starting with the base case of a single-element row. It uses the same Pascal's rule for constructing
    each row, where each element is the sum of the two elements directly above it from the previous row.

    Args:
    ----
        num_rows (int): The number of rows of Pascal's Triangle to generate.
        triangle (List[List[int]], optional): The current state of the triangle being constructed. This
            parameter is used for recursive calls and should not be provided for the initial call.

    Returns:
    -------
        List[List[int]]: A list of lists, where each sublist represents a row of Pascal's Triangle.

    Examples:
    --------
        >>> generate_recursive(5)
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]

    """
    if num_rows == 0:
        return []
    if num_rows == 1:
        return [[1]]
    if triangle is None:  # Initialize triangle on first call
        triangle = [[1]]
    if len(triangle) < num_rows:
        prev_row = triangle[-1]
        curr_row = [1] + [prev_row[i] + prev_row[i + 1] for i in range(len(prev_row) - 1)] + [1]
        triangle.append(curr_row)
        generate_recursive(num_rows, triangle)
    return triangle


print(generate_iterative(6))
