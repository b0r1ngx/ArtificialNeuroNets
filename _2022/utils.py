from torch import tensor
from random import randrange


def prepare_data(data, window_size=10, delay=0, predict_over=10):
    result = []
    for i in range(len(data) - window_size - predict_over - delay):
        y_from, y_to = i, i + window_size
        y_predict = i + window_size + delay + predict_over
        # if you want to predict a range of values, add +1 to loop range, and use it:
        # y_predict_from, y_predict_to = i + window_size + delay, i + window_size + delay + predict_over
        result.append([
            tensor(data[y_from:y_to]),
            tensor([data[y_predict]])
        ])
    return result


def train_test_split(data: list, train=0.8, test=0.2):
    if train != 0.8:
        test = 1 - train
    else:
        train = 1 - test

    assert train + test == 1
    train_size = int(len(data) * train)
    return data[:train_size], data[train_size:]


def flat_data_between_minus_one_and_one(data, window, memory_capacity):
    """ Rule of transformation is:
        data = [0, 5, 3, 1]
        flat = [1, -1, -1]
    """
    flat = []
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            flat.append(1)
        else:
            flat.append(-1)

    autoregression_source = data[1:]
    original = [autoregression_source[window * i:window * (i + 1)]
                for i in range(len(autoregression_source) // window)]
    flatten = [flat[window * i:window * (i + 1)]
                for i in range(len(flat) // window)]

    return {
        'window': window,
        'memory_capacity': memory_capacity,
        'original': original[:memory_capacity],
        'flatten': flatten[:memory_capacity]
    }


def randomly_invert_part_of_data(data, part=0.5):
    data = data.copy()
    for i in range(int(len(data) * part)):
        data[randrange(0, len(data))] *= -1
    return data


def get_amount_of_difference_between_iterable(a, b):
    assert len(a) == len(b), \
        f'arrays sizes not equal: {len(a)} != {len(b)}'

    difference = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            difference += 1
    return difference


def normalize_data_between_a_b(data, min, max, a=-1, b=1) -> list[float]:
    """ to normalize in [0, n], use a=0, b=n
    https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
    result = []
    for i in range(len(data)):
        result.append(
            (b - a)
            * (data[i] - min)
            / (max - min)
            + a
        )
    return result


def denormalize_data(normalized_data, min, max, a=-1, b=1) -> list[float]:
    """ Precision - 14 digits after dot - tested,
        (It must be 100%, but anyway think that restriction is exists)

        when test on 10k data, it fails at 14-15 digit
        (also sometimes more symbols after dot when denormalize appear)
        - so of this behavior, I add round() to generating data (AR2model, and here as well)
    """
    result = []
    for i in range(len(normalized_data)):
        result.append(
            round((normalized_data[i] - a)
                  * (max - min)
                  / (b - a)
                  + min, 13)
        )
    return result


def min_max(data: list):
    """https://stackoverflow.com/questions/12200580/numpy-function-for-simultaneous-max-and-min"""
    n = len(data)
    odd = n % 2
    if not odd:
        n -= 1
    max_val = min_val = data[0]

    i = 1
    while i < n:
        x = data[i]
        y = data[i + 1]
        if x > y:
            x, y = y, x

        min_val = min(x, min_val)
        max_val = max(y, max_val)
        i += 2

    if not odd:
        x = data[n]
        min_val = min(x, min_val)
        max_val = max(x, max_val)

    return min_val, max_val
