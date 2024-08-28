def serialize(matrix: list[list[int]]) -> str:
    rows = [" ".join(map(str, row)) for row in matrix]
    return "\n".join(rows)
