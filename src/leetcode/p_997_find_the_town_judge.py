"""997. Find the Town Judge
Easy

In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

The town judge trusts nobody.
Everybody (except for the town judge) trusts the town judge.
There is exactly one person that satisfies properties 1 and 2.
You are given an array trust where trust[i] = [a_i, b_i] representing that
the person labeled a_i trusts the person labeled b_i.
If a trust relationship does not exist in trust array,
then such a trust relationship does not exist.

Return the label of the town judge if the town judge exists and can be identified,
or return -1 otherwise.



Example 1:
Input: n = 2, trust = [[1,2]]
Output: 2

Example 2:
Input: n = 3, trust = [[1,3],[2,3]]
Output: 3

Example 3:
Input: n = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1

Constraints:
1 <= n <= 1000
0 <= trust.length <= 104
trust[i].length == 2
All the pairs of trust are unique.
ai != bi
1 <= ai, bi <= n
"""

from collections import defaultdict


class FindTownJudge:
    def solve(self, n: int, trust: list[list[int]]) -> int:
        if len(trust) == 0:
            return 1 if n == 1 else -1

        trusting = defaultdict(int)
        trusted = defaultdict(int)
        candidate = -1
        for [a, b] in trust:
            trusting[a] += 1
            trusted[b] += 1
            if candidate == -1 or trusted[b] > trusted[candidate]:
                candidate = b

        return candidate if candidate not in trusting and trusted[candidate] == n - 1 else -1

    def solve_graph(self, n: int, trust: list[list[int]]) -> int:
        """1. Initialization: Initialize two vectors, in and out, to store the in-degree and out-degree of each person.

        2. Counting Trust Relationships: Iterate through each trust relationship, incrementing the out count
        for the truster and the in count for the trustee.

        3. Finding the Judge: Iterate through each person, checking if they have an in-degree of N - 1 and
        an out-degree of 0, indicating they are trusted by everyone and trust no one.

        4. Return Value: Return the index of the judge if found, otherwise return -1.

        5. Overall Functionality: The function identifies the judge within a community based on trust relationships,
        leveraging counts of in-degree and out-degree.
        """
        in_degree = [0] * (n + 1)
        out_degree = [0] * (n + 1)
        for [a, b] in trust:
            out_degree[a] += 1
            in_degree[b] += 1
        for i in range(1, n + 1):
            if in_degree[i] == n - 1 and out_degree[i] == 0:
                return i
        return -1
