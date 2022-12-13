from torch import tensor

from _2022.AR2model import generate_model_data_with_plt


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


def flat_data_between_minus_one_and_one(data):
    """ Rule of transformation is:
        data = [0, 5, 3, 1]
        result = [1, -1, -1]
    """
    result = []
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            result.append(1)
        else:
            result.append(-1)

    return result


def randomly_invert_data(data, part=0.5):
    # for _ in range
    return


def normalize_data_between_zero_and_n(data, n=1) -> list[float]:
    min, max = min_max(data)
    result = []
    for i in range(len(data)):
        result.append((data[i] - min) / (max - min) * n)
    return result


def normalize_data_between_a_b(data, min, max, a=-1, b=1) -> list[float]:
    """https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1"""
    result = []
    for i in range(len(data)):
        result.append((b - a) * (data[i] - min) / (max - min) + a)
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


def train_test_split(data: list, train=0.8, test=0.2):
    if train != 0.8:
        test = 1 - train
    else:
        train = 1 - test

    assert train + test == 1
    train_size = int(len(data) * train)
    return data[:train_size], data[train_size:]


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
