"""
https://leetcode.com/problems/evaluate-division/description

399. Evaluate Division

Medium

You are given an array of variable pairs equations and an array of real numbers values,
where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i].
Each Ai or Bi is a string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents
the jth query where you must find the answer for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Note: The input is always valid. You may assume that evaluating the queries
will not result in division by zero and that there is no contradiction.

Note: The variables that do not occur in the list of equations are undefined,
so the answer cannot be determined for them.



Example 1:
Input: equations = [["a","b"],["b","c"]], values = [2.0,3.0],
queries = [["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]]
Output: [6.00000,0.50000,-1.00000,1.00000,-1.00000]
Explanation:
Given: a / b = 2.0, b / c = 3.0
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ?
return: [6.0, 0.5, -1.0, 1.0, -1.0 ]
note: x is undefined => -1.0

Example 2:
Input: equations = [["a","b"],["b","c"],["bc","cd"]], values = [1.5,2.5,5.0],
queries = [["a","c"],["c","b"],["bc","cd"],["cd","bc"]]
Output: [3.75000,0.40000,5.00000,0.20000]

Example 3:
Input: equations = [["a","b"]], values = [0.5],
queries = [["a","b"],["b","a"],["a","c"],["x","y"]]
Output: [0.50000,2.00000,-1.00000,-1.00000]


Constraints:

1 <= equations.length <= 20
equations[i].length == 2
1 <= Ai.length, Bi.length <= 5
values.length == equations.length
0.0 < values[i] <= 20.0
1 <= queries.length <= 20
queries[i].length == 2
1 <= Cj.length, Dj.length <= 5
Ai, Bi, Cj, Dj consist of lower case English letters and digits.

Hint 1: Do you recognize this as a graph problem?
"""

from collections import defaultdict, deque


def calc_equation_bfs(equations: list[list[str]], values: list[float], queries: list[list[str]]) -> list[float]:
    graph = defaultdict(list)
    # build a weighted graph
    for idx, (a, b) in enumerate(equations):
        graph[a].append((b, values[idx]))  # src -> [tuple(target, weight)+ ]
        graph[b].append((a, 1 / values[idx]))  # the inverse path

    def _bfs(src, target):
        if src not in graph or target not in graph:
            return -1.0

        queue = deque([(src, 1.0)])  # (current node, accumulated weight)
        visited = {src}
        while queue:
            node, weight_so_far = queue.popleft()
            if node == target:
                return weight_so_far
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    queue.append((neighbor, weight_so_far * weight))
                    visited.add(neighbor)

        return -1.0

    # calculate each query as a multiplication of the path from a to b
    return [_bfs(*query) for query in queries]


def calc_equation_dfs(equations: list[list[str]], values: list[float], queries: list[list[str]]) -> list[float]:
    graph = defaultdict(list)
    # build a weighted graph
    for idx, (a, b) in enumerate(equations):
        graph[a].append([b, values[idx]])  # src -> [[target, weight]+ ]
        graph[b].append([a, 1 / values[idx]])  # the inverse path

    def _dfs(src, target):
        if src not in graph or target not in graph:
            return -1.0
        stack, visited = [(src, 1.0)], set()
        visited.add(src)
        while stack:
            node, weight_so_far = stack.pop()
            if node == target:
                return weight_so_far
            visited.add(node)
            for neighbor, weight in graph[node]:
                if neighbor not in visited:
                    stack.append((neighbor, weight_so_far * weight))
        return -1.0

    # calculate each query as a multiplication of the path from a to b
    return [_dfs(*query) for query in queries]


if __name__ == "__main__":
    equations = [["a", "b"], ["b", "c"]]
    values = [2.0, 3.0]
    queries = [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]
    print(calc_equation_bfs(equations, values, queries))
    print(calc_equation_dfs(equations, values, queries))
