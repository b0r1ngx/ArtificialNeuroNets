from prettytable import PrettyTable

from _2022.AR2model import generate_model_data_with_plt
from _2022.lab02.HopfieldNetwork import HopfieldNetwork
from _2022.utils import prepare_data, flat_data_between_minus_one_and_one, \
    randomly_invert_part_of_data, differences_between_iterable

# Generate Data
n = 500000
data = generate_model_data_with_plt(n=n + 1)
print("AR(2)-model size:", len(data))

# Initial values to start NN
modes = [True, False]  # sync, async
observation_window_sizes = [2500]
neurons_of_network = observation_window_sizes
invert_parts = [0.5]
storage_capacities = []

for window_size in observation_window_sizes:
    storage_capacities.append(int(n / window_size))
# storage_capacities = [100, 40, 25, 10, 5]

datas = []
for window in observation_window_sizes:
    for storage_capacity in storage_capacities:
        if (n / window) >= storage_capacity:
            datas.append(
                flat_data_between_minus_one_and_one(
                    data, window, storage_capacity
                )
            )

results = {}
for part in invert_parts:
    results[part] = []
