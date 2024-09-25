"""
https://leetcode.com/problems/clone-graph/description

133. Clone Graph

Medium

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed).
For example, the first node with val == 1, the second node with val == 2, and so on.
The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph.
Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1.
You must return the copy of the given node as a reference to the cloned graph.


Example 1:
Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
Output: [[2,4],[1,3],[2,4],[1,3]]
Explanation: There are 4 nodes in the graph.
1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

Example 2:
Input: adjList = [[]]
Output: [[]]
Explanation: Note that the input contains one empty list.
The graph consists of only one node with val = 1 and it does not have any neighbors.
Example 3:

Input: adjList = []
Output: []
Explanation: This an empty graph, it does not have any nodes.


Constraints:

The number of nodes in the graph is in the range [0, 100].
1 <= Node.val <= 100
Node.val is unique for each node.
There are no repeated edges and no self-loops in the graph.
The Graph is connected and all nodes can be visited starting from the given node.
"""

from collections import deque


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

    def __repr__(self) -> str:
        return f"{self.val}, {[n.val for n in self.neighbors]}"


def clone_graph_dfs(node: Node | None) -> Node | None:
    if not node:
        return None

    clones = {}  # maps old node to clones

    def _dfs_clone(node):
        if node in clones:
            return clones[node]

        clone = Node(node.val)
        clones[node] = clone
        for nei in node.neighbors:
            clone.neighbors.append(_dfs_clone(nei))
        return clone

    return _dfs_clone(node)


def clone_graph_dfs_explicit_stack(node: Node | None) -> Node | None:
    if not node:
        return None

    clones = {node: Node(node.val)}  # maps original node to its clone
    stack = [node]  # use a stack to perform iterative DFS

    while stack:
        curr = stack.pop()
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                stack.append(neighbor)
            # Append the cloned neighbor to the current node's clone neighbors
            clones[curr].neighbors.append(clones[neighbor])

    return clones[node]


def clone_graph_bfs(node: Node | None) -> Node | None:
    if not node:
        return None

    clones = {node: Node(node.val)}  # map original nodes to their clones
    queue = deque([node])  # use a queue for BFS

    while queue:
        curr = queue.popleft()
        for neighbor in curr.neighbors:
            if neighbor not in clones:
                clones[neighbor] = Node(neighbor.val)
                queue.append(neighbor)
            # Append the cloned neighbor to the current node's clone neighbors
            clones[curr].neighbors.append(clones[neighbor])

    return clones[node]


def build_graph(adj_list: list[list[int]]) -> Node:
    if not adj_list:
        return None

    nodes = {}
    for idx in range(len(adj_list)):
        nodes[idx + 1] = Node(idx + 1)

    for idx, neighbors in enumerate(adj_list):
        curr = nodes[idx + 1]
        for neighbor in neighbors:
            curr.neighbors.append(nodes[neighbor])
    return nodes[1] if nodes else None


def are_graphs_equal(node1: Node, node2: Node, visited) -> bool:
    if not node1 and not node2 or node1 == node2:
        return True
    if not node1 and node2 or node1 and not node2:
        return False
    if node1.val != node2.val:
        return False
    visited.add(node1.val)
    if sorted([n.val for n in node1.neighbors]) != sorted([n.val for n in node2.neighbors]):
        return False
    for nei1, nei2 in zip(node1.neighbors, node2.neighbors, strict=False):
        if nei1.val not in visited:
            if not are_graphs_equal(nei1, nei2, visited):
                return False
    return True


if __name__ == "__main__":
    adj_list = [[2, 4], [1, 3], [2, 4], [1, 3]]

    original_graph = build_graph(adj_list)
    print(original_graph)

    cloned_graph = clone_graph_dfs(original_graph)
    assert are_graphs_equal(original_graph, cloned_graph, set()), "The graphs are not equal!"

    cloned_graph = clone_graph_dfs_explicit_stack(original_graph)
    assert are_graphs_equal(original_graph, cloned_graph, set()), "The graphs are not equal!"

    cloned_graph = clone_graph_bfs(original_graph)
    assert are_graphs_equal(original_graph, cloned_graph, set()), "The graphs are not equal!"
