from numpy import array, add
import torch
import matplotlib.pyplot as plt

from _2022.AR2model import generate_model_data_with_plt
from _2022.lab03.SelfOrganizingMap import SelfOrganizingMap

# Generate Data
n = 1350
data = generate_model_data_with_plt(n=n)
print("AR(2)-model size:", len(data))

# Initial values to start NN
window_size = 50
number_of_patterns = int(n / window_size)

# Training inputs for 10 patterns from AR(2)-model

patterns = array(
    [data[window_size * i : window_size * (i + 1)]
     for i in range(number_of_patterns)]
)

templates = [f'{i}' for i in range(number_of_patterns)]


def plot_data(data):
    for i in range(len(data)):
        plt.title(f'{i} window')
        plt.plot(data[i])
        plt.show()


plot_data(patterns)

data = []
for j in range(patterns.shape[0]):
    data.append(torch.FloatTensor(patterns[j, :]))

m = 10
n = 10

iter = 100
som = SelfOrganizingMap(m, n, window_size, iter)

print(f'Train an m*n SOM with {iter} iterations (each vector one by one)')
for i in range(iter):
    for j in range(len(data)):
        som(data[j], i)

print('Store a centroid grid')
centroid_grid = [[] for i in range(m)]
weights = som.get_weights()
locations = som.get_locations()
for j, loc in enumerate(locations):
    centroid_grid[loc[0]].append(weights[j].numpy())

print('Map patterns to their closest neurons')
mapped = som.map_vects(torch.Tensor(patterns))

plt.imshow(add.outer(range(m), range(n)) % 2)
plt.title('Patterns SOM')
for j, m in enumerate(mapped):
    plt.text(m[1], m[0], templates[j], ha='center', va='center',
             bbox=dict(facecolor='white', alpha=0.5, lw=0))
plt.show()
