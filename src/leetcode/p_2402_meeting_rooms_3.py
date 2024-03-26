"""
https://leetcode.com/problems/meeting-rooms-iii/description

2402. Meeting Rooms III
Hard

You are given an integer n. There are n rooms numbered from 0 to n - 1.

You are given a 2D integer array meetings where meetings[i] = [starti, endi] means
that a meeting will be held during the half-closed time interval [starti, endi).
All the values of starti are unique.

Meetings are allocated to rooms in the following manner:

Each meeting will take place in the unused room with the lowest number.
If there are no available rooms, the meeting will be delayed until a room becomes free.
The delayed meeting should have the same duration as the original meeting.
When a room becomes unused, meetings that have an earlier original start time should be given the room.
Return the number of the room that held the most meetings. If there are multiple rooms,
return the room with the lowest number.

A half-closed interval [a, b) is the interval between a and b including a and not including b.

Example 1:
Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
Output: 0
Explanation:
- At time 0, both rooms are not being used. The first meeting starts in room 0.
- At time 1, only room 1 is not being used. The second meeting starts in room 1.
- At time 2, both rooms are being used. The third meeting is delayed.
- At time 3, both rooms are being used. The fourth meeting is delayed.
- At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
- At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
Both rooms 0 and 1 held 2 meetings, so we return 0.

Example 2:
Input: n = 3, meetings = [[1,20],[2,10],[3,5],[4,9],[6,8]]
Output: 1
Explanation:
- At time 1, all three rooms are not being used. The first meeting starts in room 0.
- At time 2, rooms 1 and 2 are not being used. The second meeting starts in room 1.
- At time 3, only room 2 is not being used. The third meeting starts in room 2.
- At time 4, all three rooms are being used. The fourth meeting is delayed.
- At time 5, the meeting in room 2 finishes. The fourth meeting starts in room 2 for the time period [5,10).
- At time 6, all three rooms are being used. The fifth meeting is delayed.
- At time 10, the meetings in rooms 1 and 2 finish. The fifth meeting starts in room 1 for the time period [10,12).
Room 0 held 1 meeting while rooms 1 and 2 each held 2 meetings, so we return 1.


Constraints:

1 <= n <= 100
1 <= meetings.length <= 105
meetings[i].length == 2
0 <= starti < endi <= 5 * 105
All the values of starti are unique.
"""

from heapq import heapify, heappop, heappush
from typing import List


class MostBookedMeetingRoom3:
    def solve_sorting_and_counting(self, n: int, meetings: List[List[int]]) -> int:
        """
        Approach 1: Sorting and Counting
        Intuition
        To tackle this problem, we first observe that the meetings are allocated to rooms based on two primary rules.
        The first rule dictates that each meeting is assigned to the unused room with the lowest number.
        This implies a sequential allocation strategy, ensuring that meetings are placed in rooms in ascending order.
        The second rule comes into play when there are no available rooms; in such cases, the meeting is delayed until
        a room becomes free, and when a room becomes unused, meetings with earlier original start times take precedence.

        We can employ a systematic approach to implement these rules efficiently.
        We initialize two arrays: room_availability_time and meeting_count.
        The former tracks the availability time of each room, while the latter
        records the number of meetings held in each room.

        We iterate through the sorted meetings(sorted by start time), adhering to the rule that meetings
        should be allocated based on their start times. Sorting the meetings based on their start times is
        crucial to effectively implement Rule 3, which states that when a room becomes unused,
        meetings with an earlier original start time should be given priority for that room.
        Consider a situation where meetings are not sorted, and the algorithm encounters a scenario where
        a room becomes available after hosting a meeting. Without the sorting, the algorithm might select the
        next meeting arbitrarily, possibly one with a later original start time, thus violating Rule 3.

        For each meeting, we identify the room with the earliest availability using a nested loop.

        * If we find an available room: The currently selected meeting is allocated to that room,
        and the room's availability time is updated. Since we iterate over the N rooms in sequential order,
        we are guaranteed to identify the available room with the lowest index first.
        This update involves assigning the end time of the currently selected meeting as the new availability time
        for the room. This adjustment is made because the room can only be utilized for the next meeting after the
        currently assigned meeting is finished.
        * If we don't find an available room: we must search for the room that will become available soonest.
        Therefore, we are seeking the room with the earliest available time. The duration of the currently
        selected meeting is then added to the availability time of this identified room. This ensures that
        the delayed meeting has the same duration as the original meeting and updates the room's availability time
         accordingly.
        Throughout the process, we keep track of meeting counts in each room.
        Finally, we identify the room that held the most meetings and, in the case of a tie,
        select the room with the lowest number.
        """

        next_available_start = [0] * n
        meetings_count = [0] * n

        for start, end in sorted(meetings):
            found_available_room = False
            earliest_room = 0
            for room in range(n):
                if next_available_start[room] <= start:
                    # mark room unavailable for the period of m
                    # and increase the meeting counter
                    found_available_room = True
                    next_available_start[room] = end
                    meetings_count[room] += 1
                    break

                if next_available_start[earliest_room] > next_available_start[room]:
                    earliest_room = room

            if not found_available_room:
                # use the room with the earliest available time
                next_available_start[earliest_room] += end - start  # add duration of m
                meetings_count[earliest_room] += 1
        return meetings_count.index(max(meetings_count))  # index of the first occurrence max in the list

    def solve_with_heap(self, n: int, meetings: List[List[int]]) -> int:
        """
        Approach 2: Sorting, Counting using Priority Queues
        Intuition
        In the preceding solution, the iteration over all N rooms occurs within the nested loop,
        resulting in an overall time complexity of O(M⋅N) for the for loop.
        To enhance efficiency we must explore avenues for optimization.
        We need to devise a method to obtain the next available room without the necessity of iterating
        over all N rooms. To do this we can maintain two crucial structures: unused_rooms and used_rooms.
        These structures are essentially priority queues or heaps, with unused_rooms representing
        available rooms sorted by room number, and used_rooms storing rooms in use along with
        the time they become available again.

        We start by initialization of unused_rooms as a priority queue containing all room numbers and
        used_rooms as an empty priority queue.

        unused_rooms is ordered in ascending order according to room numbers. This arrangement guarantees
        that when an element is popped from this, it returns the unused room with the lowest number.
        This is important to follow rule 1, which states that each meeting will take place in the unused room
        with the lowest number.

        used_rooms is a priority queue that contains elements in the form of {room_availability_time, room_number}.
        Here, room_availability_time signifies the time at which this room becomes unused.
        This priority queue is ordered in ascending order based on both room_availability_time and room_number.
        This ensures that when an element is popped from it, the room returned is the one that becomes unused earliest.
        This assists in adhering to rules 2 and 3 while allocating a meeting to the room that becomes unused earliest
        when all rooms are currently in use.

        Then we proceed to iterate through the meetings after sorting them based on their start times,
        adhering to the rule that meetings should be allocated based on their start times. Within this loop,
        a cascading series of decisions unfolds to handle various scenarios.

        When iterating through meetings we first manage the release of rooms that have become unused.
        We iterate through used_rooms, popping rooms from the heap if their availability time is earlier
        than or equal to the start time of the current meeting. Released rooms are then pushed into unused_rooms.

        Subsequently, we check if there are available rooms in unused_rooms. If so, the room with the lowest number
        is assigned to the current meeting. This follows the principle of allocating meetings to the unused room with
        the lowest number.

        In the event that no rooms are available in unused_rooms, we resort to delaying the current meeting.
        We find the room with the earliest availability time (derived from the first item in used_rooms.)
        We then adjust the availability time of this room based on the duration of the delayed meeting,
        and push the room back into used_rooms. This ensures that meetings with earlier original start times
        are given priority when rooms become available and delayed meetings have the same duration
        as the original meeting.

        Throughout this process, a crucial aspect is tracking of the count of meetings held in each room using
        the meeting_count array. This array is instrumental in determining the room that hosted the most meetings.
        After we have selected the room that hosts the meeting, we increment the count of meetings that occurred
        in that room.

        Finally, we identify the room that held the most meetings and, in the case of a tie, select the room with
        the lowest number.
        """

        # Priority queue containing tuples (time, room i) to keep next available time slot for room at index i
        busy_rooms = []
        available_rooms = list(range(n))
        meeting_count = [0] * n
        heapify(available_rooms)

        for start, end in sorted(meetings):
            # Start by checking if busy rooms can be made available again
            while busy_rooms and busy_rooms[0][0] <= start:
                # The meeting at the first room has finished,
                # and so, we can move this room to the available heap/queue
                _, room = heappop(busy_rooms)
                heappush(available_rooms, room)
            # Now, allocate a room for the current meeting
            if available_rooms:
                room = heappop(available_rooms)
                heappush(busy_rooms, (end, room))
            else:
                # get the earliest possible available room (which is at the top of the heap)
                next_time_slot, room = heappop(busy_rooms)
                duration = end - start
                # update the next availability time slot for this room
                heappush(busy_rooms, (next_time_slot + duration, room))
            meeting_count[room] += 1
        return meeting_count.index(max(meeting_count))
