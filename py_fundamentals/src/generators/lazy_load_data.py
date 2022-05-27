"""
From the book: High performance python, 2nd Ed.

Given a datafile of the form “timestamp, value,”
find days whose values differ from normal distribution.
We start by writing the code that will read the file, line by line,
and output each line’s value as a Python object.
We will also create a read_fake_data generator to generate fake data
that we can test our algorithms with. For this function we still
take the argument filename, so as to have the same function signature
as read_data; however, we will simply disregard it.
These two functions are indeed lazily evaluated—we read the next
line in the file, or generate new fake data, only when the next()
function is called.


"""
from datetime import datetime
from itertools import count, filterfalse, groupby, islice
from random import normalvariate, randint

from scipy.stats import normaltest


def read_data(filename):
    with open(filename) as fd:
        for line in fd:
            data = line.strip().split(",")
            timestamp, value = map(int, data)
            yield datetime.fromtimestamp(timestamp), value


def read_fake_data():
    for timestamp in count():
        #  We insert an anomalous data point approximately once a week
        if randint(0, 7 * 60 * 60 * 24 - 1) == 1:
            value = normalvariate(0, 1)
        else:
            value = 100
        yield datetime.fromtimestamp(timestamp), value


def groupby_day(iterable):
    """
    Outputs a generator that groups of data that occur in the same day.


    The limitation is that groups will be formed only for data that is sequential.
    So if we had the input A A A A B B A A and had groupby group by the letter,
    we would get three groups: (A, [A, A, A, A]), (B, [B, B]), and (A, [A, A])
    """

    def key(row):
        return row[0].day

    for day, data_group in groupby(iterable, key):
        # Cast data_group into a list because
        # the normaltest function requires an array-like object.
        # Can be avoided by implementing an on-line version
        # of Welford’s online averaging algorithm to calculate
        # the skew and kurtosis of the numbers
        yield list(data_group)


def groupby_window(data, window_size=3600):
    window = tuple(islice(data, window_size))
    for item in data:
        yield window
        window = window[1:] + (item,)


def is_normal(data, threshold=1e-3):
    """
    Given one group of data, returns whether it follows the normal distribution
    """
    _, values = zip(*data)
    k2, p_value = normaltest(values)
    if p_value < threshold:
        return False
    return True


def filter_anomalous_groups(data):
    """
    Filter down the full dataset only to inputs that don’t pass the test
    """
    yield from filterfalse(is_normal, data)


def filter_anomalous_data(data):
    data_group = groupby_day(data)
    yield from filter_anomalous_groups(data_group)


if __name__ == "__main__":
    # Chaining the generators
    # This method allows us to get the list of days that are anomalous
    # without having to load the entire dataset. Only enough data is read
    # to generate the first five anomalies. Additionally, the anomaly_generator
    # object can be read further to continue retrieving anomalous data
    data = read_fake_data()
    anomaly_generator = filter_anomalous_data(data)
    first_five_anomalies = islice(anomaly_generator, 5)

    for data_anomaly in first_five_anomalies:
        start_date = data_anomaly[0][0]
        end_date = data_anomaly[-1][0]
        print(f"Anomaly from {start_date} - {end_date}")
