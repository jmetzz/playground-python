"""
https://leetcode.com/problems/course-schedule-ii/description
210. Course Schedule II

Medium

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1.
You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates
that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to
first take course 1.
Return the ordering of courses you should take to finish all courses.
If there are many valid answers, return any of them.
If it is impossible to finish all courses, return an empty array.


Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1
you should have finished course 0. So the correct course order is [0,1].

Example 2:
Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
Output: [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3
you should have finished both courses 1 and 2. Both courses 1 and 2
should be taken after you finished course 0.
So one correct course order is [0,1,2,3].
Another correct ordering is [0,2,1,3].

Example 3:
Input: numCourses = 1, prerequisites = []
Output: [0]


Constraints:

1 <= numCourses <= 2000
0 <= prerequisites.length <= numCourses * (numCourses - 1)
prerequisites[i].length == 2
0 <= ai, bi < numCourses
ai != bi
All the pairs [ai, bi] are distinct.

Hint 1: This problem is equivalent to finding the topological order in a directed graph.
        If a cycle exists, no topological ordering exists and therefore it will be impossible
        to take all courses.
Hint 2: Topological Sort via DFS - A great video tutorial (21 minutes) on Coursera
        explaining the basic concepts of Topological Sort.
        Link https://www.youtube.com/watch?v=ozso3xxkVGU
Hint 3: Topological sort could also be done via BFS.

"""


def find_order(num_courses: int, prerequisites: list[list[int]]) -> list[int]:
    graph = {course: [] for course in range(num_courses)}

    for course, pre_rec in prerequisites:
        graph[course].append(pre_rec)

    visited, cycle = set(), set()
    output_sequence = []

    def _dfs(course) -> bool:
        if course in cycle:
            return False
        if course in visited:
            return True

        cycle.add(course)
        for pre in graph[course]:
            if _dfs(pre) is False:
                return False

        cycle.remove(course)
        visited.add(course)
        output_sequence.append(course)
        return True

    for node in range(num_courses):
        if _dfs(node) is False:
            return []
    return output_sequence


if __name__ == "__main__":
    print(find_order(1, []))
    print(find_order(2, [[1, 0]]))
    print(find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
