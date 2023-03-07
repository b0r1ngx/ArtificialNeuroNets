from minisom import MiniSom
from numpy import array, add
import torch
import matplotlib.pyplot as plt

from _2022.lab03.SelfOrganizingMap import SelfOrganizingMap
from _2022.AR2model import generate_model_data_with_plt

# Generate Data
n = 250
data = generate_model_data_with_plt(n=n, const=-0.5)
data1 = generate_model_data_with_plt(n=n, const=0.5)
print("AR(2)-model size:", len(data))

# Initial values to start NN
window_size = 250
number_of_patterns = int(2 * n / window_size)

# Training inputs for 10 patterns from AR(2)-model

patterns = array(
    [data[window_size * i: window_size * (i + 1)]
     for i in range(number_of_patterns)]
)

templates = [f'{i}' for i in range(number_of_patterns)]


def plot_data(data):
    for i in range(len(data)):
        plt.title(f'{i} window')
        plt.plot(data[i])
        plt.show()


plot_data(patterns)

m = 2
n = 1
iter = 100

print('Create and train MiniSom')
som = MiniSom(m, n, window_size, sigma=1, learning_rate=0.1,
              neighborhood_function='triangle')
som.train(patterns, iter)

winners = [som.winner(data) for data in patterns]


def plot_winners(winners, color, title):
    plt.figure()
    plt.minorticks_on()
    plt.grid(which='major', linewidth=1)
    plt.grid(which='minor', linewidth=1, linestyle=':')

    plt.title(title)
    for i in range(len(winners)):
        plt.scatter(winners[i][0], winners[i][1], marker=f"${str(i)}$", color=color, s=300)
    plt.show()


plot_winners(winners, color='b', title='Выигрывающие нейроны - тренировочные данные')

print(f'Create and train SOM({m}x{n}) with {iter} iterations (each vector one by one)')
data = []
for j in range(patterns.shape[0]):
    data.append(torch.FloatTensor(patterns[j, :]))

som = SelfOrganizingMap(m, n, window_size, iter, sigma=1)
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

test1 = generate_model_data_with_plt(n=250)