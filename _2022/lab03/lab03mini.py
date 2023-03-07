# Generate Data
from matplotlib import pyplot as plt
from minisom import MiniSom
from numpy import array

from _2022.AR2model import generate_model_data_with_plt

n = 5000
data = generate_model_data_with_plt(n=n)
print("AR(2)-model size:", len(data))

# Initial values to start NN
window_size = 250
number_of_patterns = int(n / window_size)

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

m = 6
n = 6
iter = 100

som = MiniSom(m, n, window_size, sigma=4, learning_rate=0.1, neighborhood_function='triangle')
som.train(patterns, iter)

winners = [som.winner(data) for data in patterns]


def plot_winners(winners, color, description):
    plt.figure()
    plt.minorticks_on()
    plt.grid(which='major', linewidth=1)
    plt.grid(which='minor', linewidth=1, linestyle=':')

    plt.title(description)
    for i in range(len(winners)):
        plt.scatter(winners[i][0], winners[i][1], marker=f"${str(i)}$", color=color, s=300)
    plt.show()


print(winners)
plot_winners(winners, color='b', description='Выигрывающие нейроны - тренировочные данные')