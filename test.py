from _2022.AR2model import generate_model_data_with_plt
from _2022.utils import flat_data_between_minus_one_and_one

window = 50
capacity = 2

# data = generate_model_data_with_plt()
data1 = generate_model_data_with_plt(n=101)

# d = flat_data_between_minus_one_and_one(data, window, capacity)
d1 = flat_data_between_minus_one_and_one(data1, window, capacity)

# print(d)
# print(data[:window*capacity])
# print(len(d['original'][0]))
# print(len(d['original'][1]))
# print(len(d['flatten'][0]))
# print(len(d['flatten'][1]))
print(d1)
print(data1)
print(data1[:window*capacity])
print(len(d1['original'][0]))
print(len(d1['original'][1]))
print(len(d1['flatten'][0]))
print(len(d1['flatten'][1]))
