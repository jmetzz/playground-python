"""
207. Course Schedule
Medium

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
You are given an array prerequisites where
    prerequisites[i] = [ai, bi]
indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.


Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: true
Explanation: There are a total of 2 courses to take.
To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take.
To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1.
So it is impossible.


Constraints:

1 <= numCourses <= 2000
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
All the pairs prerequisites[i] are unique.
"""

from collections import deque
from typing import Dict, List, Tuple


def build_dependency_graph(num_items: int, dependency_list: List[List[int]]) -> Tuple[Dict, List]:
    graph = {course: [] for course in range(num_items)}
    indegree = [0] * num_items
    for course, dependency in dependency_list:
        graph[dependency].append(course)
        indegree[course] += 1
    return graph, indegree


def topological_sort_dfs(graph: Dict[int, List]) -> List[int]:
    _path_stack, _order = [], []
    _visited = set()

    def _dfs(vertex) -> None:
        _path_stack.append(vertex)  # accumulate the vertex in the current path until backtrack
        for neighbor in graph[vertex]:
            if neighbor not in _visited:
                _visited.add(neighbor)
                _dfs(neighbor)
        # after the end of the path is reached,
        # backtrack and move the top of the path stack
        # to the topological order
        _order.append(_path_stack.pop())

    for v in graph:
        if v not in _visited:
            _visited.add(v)
            _dfs(v)
    return _order[::-1]


def can_finish_dfs_1(num_courses: int, prerequisites: List[List[int]]) -> bool:
    graph, _ = build_dependency_graph(num_courses, prerequisites)
    return len(topological_sort_dfs(graph)) == num_courses


def can_finish_dfs_2(num_courses: int, prerequisites: List[List[int]]) -> bool:
    graph, _ = build_dependency_graph(num_courses, prerequisites)
    _path_stack = set()
    _visited = set()

    def _dfs(vertex) -> bool:
        _path_stack.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor in _path_stack:
                return False
            if neighbor not in _visited:
                _visited.add(neighbor)
                if _dfs(neighbor) is False:
                    return False
        _path_stack.remove(vertex)
        return True

    for course in range(num_courses):
        if course not in _visited:
            _visited.add(course)
            if _dfs(course) is False:
                return False
    return True


def topological_sort_bfs(graph: Dict[int, List], in_degree: List[int]) -> List[int]:
    # remember: bfs uses a queue instead of stack
    # We start with nodes that have no pre-requisites.
    # Thus, add the que processing queue all vertices without dependencies
    # ie, indegree equal to zero
    num_courses = len(in_degree)
    satisfied_dependencies = deque([course for course in range(num_courses) if in_degree[course] == 0])
    traverse_order = []
    while satisfied_dependencies:
        # process each vertex in the queue,
        # when visiting the verted, we "remove it"
        # from the graph once visited by decreasing
        # the indegree of all its neighbors to simmulate
        # the current vertex removal
        vertex = satisfied_dependencies.popleft()
        for neighbor in graph[vertex]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                # all pre-requisites of neighbor are satisfied
                satisfied_dependencies.append(neighbor)
        traverse_order.append(vertex)
    return traverse_order


def can_finish_bfs_1(num_courses: int, prerequisites: List[List[int]]) -> bool:
    # represent the problem as a graph, in which vertices are courses,
    # and the directed edges represent the dependency edge, ie,
    # the source vertex depends on the list of neighbors vertices.
    # In this case, also use the indegree of each vertex as a proxy for
    # being in the graph or not. Zero indegree means
    # the vertex has no pre-requisites, and thus
    # the course can be completed.
    graph, in_degree = build_dependency_graph(num_courses, prerequisites)
    _order = topological_sort_bfs(graph, in_degree)
    return len(_order) == num_courses


def can_finish_bfs_2(num_courses: int, prerequisites: List[List[int]]) -> bool:
    graph = {course: [] for course in range(num_courses)}
    for course, dependency in prerequisites:
        graph[dependency].append(course)

    _visited, _path_stack = set(), set()

    def _dfs(vertex) -> bool:
        _path_stack.add(vertex)
        for neighbor in graph[vertex]:
            if neighbor in _path_stack:
                return False
            if neighbor not in _visited:
                _visited.add(neighbor)
                if _dfs(neighbor) is False:
                    return False
        _path_stack.remove(vertex)
        return True

    for course in range(num_courses):
        if course not in _visited:
            _visited.add(course)
            if _dfs(course) is False:
                return False
    return True


print(can_finish_dfs_1(5, [[0, 1], [1, 2], [0, 2], [2, 3], [4, 2]]))
print("-" * 20)
print(can_finish_dfs_2(5, [[0, 1], [1, 2], [0, 2], [2, 3], [4, 2]]))
print("-" * 20)
print(can_finish_bfs_1(5, [[0, 1], [1, 2], [0, 2], [2, 3], [4, 2]]))
print("-" * 20)
print(can_finish_bfs_2(5, [[0, 1], [1, 2], [0, 2], [2, 3], [4, 2]]))
