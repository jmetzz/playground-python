import pytest

from data_structures.graphs import has_path_bfs_iterative, has_path_bfs_recursive

# Define test cases for the graph structure
# Each test case is a tuple: (graph, source, target, expected_result)
test_cases = [
    ({}, 1, 2, False),  # Empty graph
    ({1: {2}, 2: {1}}, 1, 2, True),  # Cycle
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 1, True),  # Source equals target
    ({1: {2, 3}, 2: {4}, 3: {4}, 4: set()}, 1, 4, True),  # Basic case with path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 1, 3, True),  # Path exists
    ({1: {2, 3}, 2: {5}, 3: {5}, 5: set()}, 1, 4, False),  # No path
    ({1: {2}, 2: {3}, 3: {4}, 4: set()}, 4, 1, False),  # Reverse path does not exist
]


@pytest.mark.parametrize("graph, source, target, expected", test_cases)
def test_has_path_bfs_recursive(graph, source, target, expected):
    assert has_path_bfs_recursive(graph, source, target) == expected


@pytest.mark.parametrize("graph, source, target, expected", test_cases)
def test_has_path_bfs_iterative(graph, source, target, expected):
    assert has_path_bfs_iterative(graph, source, target) == expected
