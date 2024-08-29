from typing import List


def transpose(matrix: List[List[int]], square: bool = False) -> List[List[int]]:
    """Transposes a 2D matrix (flips a matrix over its diagonal).

    The function takes a square matrix as input and returns its transpose.
    The transpose of a matrix is obtained by moving the rows to
    columns or columns to rows.

    This implementation requires the input matrix to be square
    (the number of rows equal to the number of columns).
    If a non-square matrix is passed, the function raises a ValueError.

    Args:
    ----
        matrix (List[List[int]]): A square matrix represented as a list of lists of integers.
        Each sublist represents a row in the matrix.

    Returns:
    -------
        List[List[int]]: The transposed square matrix represented as a list of lists of integers.
        Each sublist represents a row in the transposed matrix.

    Raises:
    ------
        ValueError: If the input matrix is not square
        (number of rows does not equal the number of columns).

    Example:
    -------
        >>> transpose([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

        >>> transpose([[1, 2], [3, 4], [5, 6]])
        ValueError: Expect a quadratic matrix (n==n). Given 3x2

    """
    if not matrix or matrix == []:
        return []

    if square:
        n = len(matrix)
        if len(matrix[0]) != n:
            raise ValueError(f"Expect a quadratic matrix (n==n). Given {n}x{len(matrix[0])}")

    return list(map(list, zip(*matrix)))


def serialize(matrix: list[list[int]], justify: bool = True) -> str:
    if not matrix:
        return ""
    if justify:
        col_widths = [max(len(str(item)) for item in col) for col in zip(*matrix)]
        rows = []
        for row in matrix:
            formatted_row = " | ".join(f"{str(item).rjust(width)}" for item, width in zip(row, col_widths))
            rows.append(f"| {formatted_row} |")
    else:
        rows = [" ".join(map(str, row)) for row in matrix]

    return "\n".join(rows)


def max_columns_width(matrix: list[list[int]]) -> list[int]:
    if not matrix or matrix == []:
        return []
    return [max(len(str(item)) for item in col) for col in zip(*matrix)]


if __name__ == "__main__":
    print("-" * 5)
    print(serialize(None))
    print("-" * 5)
    print(serialize([[]]))
    print("-" * 5)
    print(serialize([[1]]))
    print("-" * 5)
    print(serialize([[1, 2, 3]]))
    print("-" * 5)
    print(serialize([[1], [2], [3]]))
    print("-" * 5)
    print(serialize([[1, 2, 3], [4, 5, 6]]))
    print("-" * 5)
    print(serialize([[1, 2, 3], [4, 50, 6], [4, 5, 600]]))
    print("-" * 25)
    matrix = [[1, 2, 3], [4, 50, 6], [4, 5, 600]]
    matrix_t = transpose(matrix)
    print(serialize(matrix))
    print("-" * 25)
    print(serialize(matrix_t))
