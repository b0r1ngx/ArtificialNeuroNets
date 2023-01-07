from prettytable import PrettyTable
import matplotlib.pyplot as plt

from _2022.AR2model import ar2_model, generate_model_data_with_plt
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


def plot_data(data):
    original = data['original']
    flatten = data['flatten']
    for i in range(len(original)):
        plt.title(f'{i + 1} window - original & flatten')
        plt.plot(original[i])
        plt.plot(flatten[i])
        plt.show()


def determine_recognised_as(c, p):
    for i in range(len(c)):
        correct_window = c[i]
        if differences_between_iterable(correct_window, p) == 0:
            return i + 1
    return None


def plot_correct_n_predictions(c, p):
    errors = 0
    for i in range(len(c)):
        error = differences_between_iterable(c[i], p[i][0])
        if error > 0:
            errors += error
            recognised_as = determine_recognised_as(c, p[i][0])
            if recognised_as:
                title = f'{i + 1} window (original to predict), ' \
                        f'diffs={error}, recognised as: {recognised_as}-window'
                print(title)
                plt.title(title)
                plt.plot(c[i])
                plt.plot(p[i][0])
                plt.show()

    return errors


for data in datas:
    hopfield_nn = HopfieldNetwork()
    hopfield_nn.train_weights(data['flatten'])
    hopfield_nn.plot_weights()

    for mode in modes:
        for part in invert_parts:
            inverted_data = [
                randomly_invert_part_of_data(i, part_to_invert=part)
                for i in data['flatten']
            ]
            predictions = [
                hopfield_nn.predict([i], sync=mode)
                for i in inverted_data
            ]

            errors = plot_correct_n_predictions(data['flatten'], predictions)

            results[part].append({
                'sync': mode,
                'window': data['window'],
                'storage_capacity': data['storage_capacity'],
                'storage_capacity_limit': data['storage_capacity_limit'],
                'error': errors,
            })

# for data in datas:
#     plot_data(data)

for part in invert_parts:
    table = PrettyTable()
    table.title = f"Inverted data - {int(part * 100)}%"
    table.field_names = ['sync', 'window', 'storage used/limit', 'error']
    for res in results[part]:
        table.add_row(
            [res['sync'], res['window'],
             f"{res['storage_capacity']}/{res['storage_capacity_limit']}",
             res['error']])

    print(table)
