"""
https://leetcode.com/problems/minimum-height-trees/description

310. Minimum Height Trees
Medium

A tree is an undirected graph in which any two vertices are connected by exactly one path.
In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges
where edges[i] = [ai, bi] indicates that there is an undirected edge between
the two nodes ai and bi in the tree, you can choose any node of the tree as the root.
When you select a node x as the root, the result tree has height h.
Among all possible rooted trees, those with minimum height (i.e. min(h))
are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path
between the root and a leaf.



Example 1:
Input: n = 4, edges = [[1,0],[1,2],[1,3]]
Output: [1]
Explanation: As shown, the height of the tree is 1 when the root is the node with label 1 which is the only MHT.

Example 2:
Input: n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]]
Output: [3,4]


Constraints:

1 <= n <= 2 * 10**4
edges.length == n - 1
0 <= ai, bi < n
ai != bi
All the pairs (ai, bi) are distinct.
The given input is guaranteed to be a tree and there will be no repeated edges.

"""

from collections import defaultdict
from heapq import heappush
from typing import List


def find_min_height_trees_dfs(n: int, edges: List[List[int]]) -> List[int]:
    """
    Finds all the minimum height trees (MHTs) in an undirected graph using depth-first search (DFS).

    Note: This solution will hit Time Limit Exceeded on LeetCode for large graphs
          due to its computational complexity.

    This DFS-based solution attempts to find the height of the tree rooted at each node by exploring
    each path to its deepest leaf, but it is not optimized for larger graphs.

    Parameters:
    - n (int): The number of nodes in the graph.
    - edges (List[List[int]]): A list of undirected edges where each edge is represented by a list of two integers.

    Returns:
    - List[int]: A list of integers representing the root nodes of all the minimum height trees.

    Example:
    >>> find_min_height_trees_dfs(4, [[1, 0], [1, 2], [1, 3]])
    [1]
    """

    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def _dfs(root) -> int:
        """
        Helper function to perform DFS and find the maximum height starting from 'root'
        """
        stack = [(root, 0)]
        visited = set()
        max_height = 0
        while stack:
            node, current_height = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append((neighbor, current_height + 1))
            max_height = max(max_height, current_height)
        return max_height

    max_heights_heap = []
    for root in graph:
        h = _dfs(root)
        if not max_heights_heap or max_heights_heap[0][0] < -h:
            max_heights_heap = [(-h, root)]
        elif max_heights_heap[0][0] == -h:
            heappush(max_heights_heap, (-h, root))

    # Extract the root nodes from the heap that have the minimum height
    return [root for _, root in max_heights_heap]


def find_min_height_trees_removing_leaves(n: int, edges: List[List[int]]) -> List[int]:
    """
    Finds all the minimum height trees (MHTs) in an undirected graph by iteratively removing leaf nodes.

    This method identifies the centroids of the graph by pruning leaf nodes until only one or two nodes are left,
    which will be the root nodes of the trees with the minimum height.

    - The function first constructs a graph as an adjacency list.
    - Nodes with only one connected node (leaf nodes) are initially identified.
    - The algorithm removes leaf nodes layer by layer until the remaining graph has only one or two nodes.
    - This approach ensures that the remaining node(s) are the centroids of the graph, minimizing the height
      when used as the root of the tree.

    The solution handles edge cases such as a graph with only one node or no edges gracefully.

    Parameters:
    - n (int): The number of nodes in the graph.
    - edges (List[List[int]]): A list of undirected edges where each edge is represented by a list of two integers.

    Returns:
    - List[int]: A list of integers representing the root nodes of all the minimum height trees. The list will
      contain either one or two elements, depending on the structure of the tree.

    Example:
    >>> find_min_height_trees_removing_leaves(6, [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]])
    [3, 4]
    """
    if n == 1:
        # Single node case, the only node is trivially the centroid.
        return [0]
    if not edges:
        # No edges case, no trees can be formed.
        return []

    # Build the graph as an adjacency list
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    # leaves have indegree equals 1
    leaves = [leaf for leaf in graph if len(graph[leaf]) == 1]

    # prune the tree until we have either 1 or 2 nodes only in the graph
    remaining_nodes = n
    while remaining_nodes > 2:
        remaining_nodes -= len(leaves)
        new_leaves = []
        for leaf in leaves:
            neighbor = graph[leaf].pop()
            graph[neighbor].remove(leaf)
            if len(graph[neighbor]) == 1:
                new_leaves.append(neighbor)
        leaves = new_leaves
    return leaves


if __name__ == "__main__":
    cases = [
        (4, [[1, 0], [1, 2], [1, 3]]),
        (6, [[3, 0], [3, 1], [3, 2], [3, 4], [5, 4]]),
        (10, [[0, 3], [1, 3], [2, 3], [4, 3], [5, 3], [4, 6], [4, 7], [5, 8], [5, 9]]),
    ]
    for n, edges in cases:
        print(find_min_height_trees_dfs(n, edges))
        print(find_min_height_trees_removing_leaves(n, edges))
        print()
