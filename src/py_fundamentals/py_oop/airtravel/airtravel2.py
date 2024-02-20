"""Model for aircraft flights"""


class Flight:
    def __init__(self, number, aircraft):
        if not number[:2].isalpha():
            raise ValueError(f"No airline code in '{number}'")

        if not number[:2].isupper():
            raise ValueError(f"Invalid airline code '{number}'")

        if not number[2:].isdigit() and int(number[2:]) <= 9999:
            raise ValueError(f"Invalid route number '{number}'")

        self._number = number
        self._aircraft = aircraft

        rows, seats = self._aircraft.seating_plan()
        self._seating = [None] + [{letter: None for letter in seats} for _ in rows]

    def number(self):
        return self._number

    def airline(self):
        return self._number[:2]

    def aircraft_model(self):
        return self._aircraft.model()

    def _parse_seat(self, seat):
        """Parse a seat designator.

        Args:
            seat: A seat designator such as '12C' or '21F'.

        Returns:
            A tuple containing an integer and a string for row and seat.
        """
        row_numbers, seat_letters = self._aircraft.seating_plan()

        letter = seat[-1]
        if letter not in seat_letters:
            raise ValueError(f"Invalid seat letter {letter}")

        row_text = seat[:-1]
        try:
            row = int(row_text)
        except ValueError:
            raise ValueError(f"Invalid seat row {row_text}")  # noqa: B904

        if row not in row_numbers:
            raise ValueError(f"Invalid row number {row_numbers}")

        return row, letter

    def allocate_seat(self, seat, passenger):
        """Allocate a seat to a passenger.

        Args:
            seat: A seat designator such as '12C' or '21F'.
            passenger: The passenger name

        Raises:
            ValueError: if the seat is unavailable
        """
        row, letter = self._parse_seat(seat)

        if self._seating[row][letter] is not None:
            raise ValueError(f"Seat {seat} already occupied")

        self._seating[row][letter] = passenger

    def relocate_passenger(self, from_seat, to_seat):
        """Relocate a passenger to a different seat

        Args:
            from_seat: The existing seat designator for
                       the passenger to be moved
            to_seat: The new seat designator
        """
        from_row, from_letter = self._parse_seat(from_seat)
        if self._seating[from_row][from_letter] is None:
            raise ValueError(f"No passenger to relocate in seat {from_seat}")

        to_row, to_letter = self._parse_seat(to_seat)
        if self._seating[to_row][to_letter] is not None:
            raise ValueError(f"Seat {to_seat} already occupied")

        self._seating[to_row][to_letter] = self._seating[from_row][from_letter]
        self._seating[from_row][from_letter] = None

    def num_available_seats(self):
        return sum(sum(1 for seat in row.values() if seat is None) for row in self._seating if row is not None)

    def make_boarding_cards(self, card_printer):
        for passenger, seat in sorted(self._passenger_seats()):
            card_printer(passenger, seat, self.number(), self.aircraft_model())

    def _passenger_seats(self):
        """An iterable series of passenger seating allocations"""
        row_numbers, seat_letters = self._aircraft.seating_plan()
        for row in row_numbers:
            for letter in seat_letters:
                passenger = self._seating[row][letter]
                if passenger is not None:
                    yield passenger, f"{row}{letter}"


class Aircraft:
    def __init__(self, registration):
        self._registration = registration

    def registration(self):
        return self._registration

    def num_seats(self):
        rows, row_seats = self.seating_plan()
        return len(rows) * len(row_seats)


class AirbusA319(Aircraft):
    @staticmethod
    def model():
        return "Airbus A319"

    @staticmethod
    def seating_plan():
        return range(1, 23), "ABCDEF"


class Boeing777(Aircraft):
    @staticmethod
    def model():
        return "Boeing 777"

    @staticmethod
    def seating_plan():
        return range(1, 56), "ABCDEFGHJK"


def make_flight():
    f = Flight("BA758", AirbusA319("G-EFJT"))
    f.allocate_seat("12A", "John Doe")
    f.allocate_seat("15F", "Isaac Newton")
    f.allocate_seat("15E", "Pablo Picasso")
    f.allocate_seat("1C", "John McCarthy")
    f.allocate_seat("1D", "Richard Hickey")

    g = Flight("BA758", Boeing777("BTXX"))
    g.allocate_seat("12A", "John Doe")
    g.allocate_seat("15F", "Isaac Newton")
    g.allocate_seat("15E", "Pablo Picasso")
    g.allocate_seat("1C", "John McCarthy")
    g.allocate_seat("1D", "Richard Hickey")
    return f, g


def console_card_printer(passenger, seat, flight_number, aircraft):
    output = f"| Name: {passenger}" f"  Flight: {flight_number}" f"  Seat: {seat}" f"  Aircraft: {aircraft}" "  |"
    banner = "+" + "-" * (len(output) - 2) + "+"
    border = "|" + " " * (len(output) - 2) + "|"
    lines = [banner, border, output, border, banner]
    card = "\n".join(lines)
    print(card)
    print()
