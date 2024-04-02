"""
https://leetcode.com/problems/number-of-recent-calls/description/
933. Number of Recent Calls

Easy

You have a RecentCounter class which counts the number of recent requests within a certain time frame.

Implement the RecentCounter class:

RecentCounter() Initializes the counter with zero recent requests.
int ping(int t) Adds a new request at time t, where t represents some time in milliseconds,
and returns the number of requests that has happened in the past 3000 milliseconds
(including the new request). Specifically, return the number of requests that have
happened in the inclusive range [t - 3000, t].

It is guaranteed that every call to ping uses a strictly larger value of t than the previous call.



Example 1:
Input
["RecentCounter", "ping", "ping", "ping", "ping"]
[[], [1], [100], [3001], [3002]]
Output
[null, 1, 2, 3, 3]

Explanation
RecentCounter recentCounter = new RecentCounter();
recentCounter.ping(1);     // requests = [1], range is [-2999,1], return 1
recentCounter.ping(100);   // requests = [1, 100], range is [-2900,100], return 2
recentCounter.ping(3001);  // requests = [1, 100, 3001], range is [1,3001], return 3
recentCounter.ping(3002);  // requests = [1, 100, 3001, 3002], range is [2,3002], return 3


Constraints:

1 <= t <= 109
Each test case will call ping with strictly increasing values of t.
At most 104 calls will be made to ping.

"""

from collections import deque


class RecentCounter:
    """
    A class to count recent requests within a specified time frame.

    Attributes:
        _max_period (int): The maximum period, in milliseconds, to consider for recent requests.
        _q (collections.deque): A deque to store the timestamps of the recent requests.

    Methods:
        ping(t): Adds a new request time and returns the number of recent requests.
    """

    def __init__(self, max_time_window: int = 3000):
        """
        Initializes the RecentCounter with an empty deque and sets the
        maximum period in milliseconds. Default to 3000.
        """
        self._max_period = max_time_window
        self._q = deque()  # Initialize an empty deque to store timestamps

    def ping(self, t: int) -> int:
        """
        Records a new request time and returns the count of recent requests.

        Args:
            t (int): The timestamp of the new request, in milliseconds.

        Returns:
            int: The number of requests within the time frame of t minus max_time_window_size
        """
        self._q.append(t)
        min_value = t - self._max_period
        # Remove requests older than the maximum period
        while self._q[0] < min_value:
            self._q.popleft()
        return len(self._q)


if __name__ == "__main__":
    counter = RecentCounter()
    # values = [1, 100, 3001, 3002, 3003, 4000]

    values = [(1, 1), (100, 2), (3001, 3), (3002, 3), (3003, 4), (4000, 4), (5000, 5), (6000, 3), (8000, 3)]
    for v, _ in values:
        print(f"ping({v}) -> {counter.ping(v)}    | q: {counter._q}")

    print(f"Current counter values: {counter._q}")
