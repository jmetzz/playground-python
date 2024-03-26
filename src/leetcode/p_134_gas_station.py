"""
https://leetcode.com/problems/gas-station/description/
134. Gas Station
Medium

There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the
ith station to its next (i + 1)th station. You begin the journey with an empty tank at
one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around
the circuit once in the clockwise direction, otherwise return -1. If there exists a solution,
it is guaranteed to be unique



Example 1:
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.

Example 2:
Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.


Constraints:

n == gas.length == cost.length
1 <= n <= 105
0 <= gas[i], cost[i] <= 104
"""

from typing import List


def can_complete_circuit_all_possibilities(gas: List[int], cost: List[int]) -> int:
    num_stations = len(gas)

    def _can_traverse(start: int) -> bool:
        fuel = 0
        next_station = start
        started = False
        while next_station != start or not started:
            started = True
            fuel += gas[next_station] - cost[next_station]
            # check if we can reach next_station
            if fuel < 0:
                return False
            next_station = (next_station + 1) % num_stations
        return True

    for start_station in range(num_stations):
        if _can_traverse(start_station):
            return start_station
    return -1


def can_complete_circuit(gas: List[int], cost: List[int]) -> int:
    # base case
    if sum(gas) < sum(cost):
        return -1

    # from this point on we are sure there is a solution,
    # and according to the description, there is only 1 solution.
    # Assuming there is enough gas to cover the costs,
    # iterate through the gas stations considering each station
    # as a potential starting point for completing the circuit.
    # While traversing the stations, maintain a running total of fuel
    # (the net gas difference between the available gas and the cost)
    fuel = 0
    # choose station zero as starting point, and assume it is
    # possible to complete the circuit from here
    candidate_station = 0
    for station in range(len(gas)):
        fuel += gas[station] - cost[station]
        if fuel < 0:
            # restart from the next station
            fuel = 0
            candidate_station = station + 1
    # If the base case above were not used,
    # we would need to consider the balance of the stations
    # that appeared before our candidate:
    # prev_stations_balance = sum(gas[:candidate_station]) - sum(cost[:candidate_station])
    # if candidate_station == len(gas) or fuel + prev_stations_balance < 0:
    #     return -1
    # else:
    #     return candidate_station
    return candidate_station


print(can_complete_circuit_all_possibilities([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
print(can_complete_circuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]))
