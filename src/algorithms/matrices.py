from typing import List


def transpose(matrix: List[List[int]]) -> List[List[int]]:
    """
    Transposes a square matrix (flips a matrix over its diagonal).

    The function takes a square matrix as input and returns its transpose.
    The transpose of a matrix is obtained by moving the rows to
    columns or columns to rows.

    This implementation requires the input matrix to be square
    (the number of rows equal to the number of columns).
    If a non-square matrix is passed, the function raises a ValueError.

    Args:
        matrix (List[List[int]]): A square matrix represented as a list of lists of integers.
        Each sublist represents a row in the matrix.

    Returns:
        List[List[int]]: The transposed square matrix represented as a list of lists of integers.
        Each sublist represents a row in the transposed matrix.

    Raises:
        ValueError: If the input matrix is not square
        (number of rows does not equal the number of columns).

    Example:
        >>> transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

        >>> transpose([[1, 2], [3, 4], [5, 6]])
        ValueError: Expect a quadratic matrix (n==n). Given 3x2
    """
    if matrix == []:
        return []

    n = len(matrix)
    if len(matrix[0]) != n:
        raise ValueError(f"Expect a quadratic matrix (n==n). Given {n}x{len(matrix[0])}")

    return list(map(list, zip(*matrix)))
