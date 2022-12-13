from prettytable import PrettyTable

from _2022.AR2model import generate_model_data_with_plt
from _2022.lab02.HopfieldNetwork import HopfieldNetwork
from _2022.utils import prepare_data, flat_data_between_minus_one_and_one, \
    randomly_invert_part_of_data, get_amount_of_difference_between_iterable

hopfield_nn = HopfieldNetwork()
modes = [True, False]  # async, sync
window_sizes = (10, 50, 100, 350, 1000)
invert_parts = (.15, .3, .45, .6, .75, .9)
memory_capacity = (2, 5, 10, 35, 50, 100)

# Generate Data
data = generate_model_data_with_plt()
print("AR(2)-model size:", len(data))

datas = []
for window in window_sizes:
    for image_num in memory_capacity:
        datas.append(
            flat_data_between_minus_one_and_one(
                data, window, image_num
            )
        )

results = {}
for part in invert_parts:
    results[part] = []
