from torch import tensor


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


def prepare_linear_data(data):
    result = []
    for i in data:
        result.append()


def train_test_split(data: list, train=0.8, test=0.2):
    if train != 0.8:
        test = 1 - train
    else:
        train = 1 - test

    assert train + test == 1
    train_size = int(len(data) * train)
    return data[:train_size], data[train_size:]
