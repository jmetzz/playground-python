class StockSpanner:
    """
    A class that calculates the span of stock prices.

    The span of a stock's price today is defined as the maximum number of consecutive days
    (starting from today and going backward) for which the price of the stock was
    less than or equal to today's price.

    The intuition:
    The core idea of this solution is to use a decreasing monotonic stack to keep track of
    the prices and their corresponding spans in a way that allows for efficient calculation of
    the span for each new price.

    The stack data structure allows us to iteratively look back at previous prices
    in a LIFO (Last In, First Out) manner, which is exactly what we need when
    calculating the span (looking back at consecutive days).

    For each new price, we compare it against the prices stored in the stack.
    If the current price is greater or equal, we "absorb" these prices into
    the span of the current price, because their presence indicates consecutive days
    where the price was less than or equal to the current price.

    This method is efficient because it avoids recalculating spans for prices
    that have already been compared, and it dynamically adjusts the span for each
    new price based on the prices that came before it.

    Attributes:
        prices_stack (list of tuple): A stack used to store pairs of (price, span),
        where 'price' is the price of the stock and 'span' is the number of days
        the price has been less than or equal to the current price.
    """

    def __init__(self):
        """
        Initializes the StockSpanner with an empty prices stack.
        """
        self.prices_stack = []  # Stores pairs of (price, span)

    def next(self, price: int) -> int:
        """
        Processes the next price of the stock and returns its span.

        The function iteratively compares the current price with the prices in the stack.
        If the current price is greater or equal to the price at the top of the stack,
        it means the current price's span includes the span of the price at the stack's top.
        This process continues until a price greater than the current price is found or
        the stack is empty. This way, we efficiently calculate the span by "accumulating"
        spans of prices that are less than or equal to the current price.

        Args:
            price (int): The price of the stock for the current day.

        Returns:
            int: The span of the stock's price for the current day.
        """
        span = 1  # Account for the current day

        # Compare the current price with the prices in the stack
        while self.prices_stack and self.prices_stack[-1][0] <= price:
            _, prev_span = self.prices_stack.pop()
            span += prev_span

        # Push the current price and its calculated span onto the stack
        self.prices_stack.append((price, span))
        return span
