import torch
from torch import tensor

from group_00201.AR2model import ar2_model
from group_00201.lab01.TimeDelayNeuralNetwork import TimeDelayNeuralNetwork

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Used device: {device}")
torch.device(device)

time_delay_nn = TimeDelayNeuralNetwork()
print(time_delay_nn)

data = tensor(ar2_model())


def create_dataset(array, window_size, delay=0, pred_steps=10):
    result = []

    for i in range(len(array) - window_size - pred_steps - delay + 1):
        x_index_start, x_index_end = i, i + window_size
        y_index_start, y_index_end = i + window_size + delay, i + window_size + delay + pred_steps

        result.append([
            array[x_index_start:x_index_end],
            array[y_index_start:y_index_end]
        ])

    # result = tensor(result)
    return result


delays = (0, 10, 100, 1000)
window_sizes = (1, 10, 50, 100)

current_data = create_dataset(data, 10)
print(current_data) # todo: understand what we need to put to our tdnn
# out = time_delay_nn(current_data)
# print(out)
