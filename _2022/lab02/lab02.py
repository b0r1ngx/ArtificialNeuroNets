from matplotlib import pyplot as plt

from _2022.AR2model import generate_model_data_with_plt
from _2022.lab02.HopfieldNetwork import HopfieldNetwork
from _2022.utils import prepare_data, train_test_split

input_size = 10

hopfield_nn = HopfieldNetwork()
print(hopfield_nn)

window_sizes = (1, 10, 50, 100)
delays = (0, 10, 100, 1000)
memory_capacity = (2, 5, 10, 50, 100)

# Generate Data
data = generate_model_data_with_plt(n=10000)
print("Size of all data from ar2_model:", len(data))
current_data = prepare_data(data, window_size=input_size)

size = len(current_data)
batch_size = len(current_data[0][0])
print("Size after split_data:", size)
print("Batch size:", batch_size)

train_data, test_data = train_test_split(current_data, train=0.5)
print("Sizes of data, after train_test_split:", len(train_data), len(test_data))
