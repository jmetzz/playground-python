import pytest

from leetcode.p_2402_meeting_rooms_3 import MostBookedMeetingRoom3


@pytest.mark.parametrize("test_input_n, test_input_meetings,  expected",
                         [
                             (2, [[0, 10], [1, 5], [2, 7], [3, 4]], 0),
                             (3, [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]], 1),
                             (3, [[1, 5], [3, 10], [5, 18], [7, 13], [8, 15], [9, 12]], 0),
                         ])
def test_most_booked_meeting_room_counting(test_input_n, test_input_meetings, expected):
    assert MostBookedMeetingRoom3().solve_sorting_and_counting(test_input_n, test_input_meetings) == expected


@pytest.mark.parametrize("test_input_n, test_input_meetings,  expected",
                         [
                             (2, [[0, 10], [1, 5], [2, 7], [3, 4]], 0),
                             (3, [[1, 20], [2, 10], [3, 5], [4, 9], [6, 8]], 1),
                             (3, [[1, 5], [3, 10], [5, 18], [7, 13], [8, 15], [9, 12]], 0),
                         ])
def test_most_booked_meeting_room_heap(test_input_n, test_input_meetings, expected):
    assert MostBookedMeetingRoom3().solve_with_heap(test_input_n, test_input_meetings) == expected
