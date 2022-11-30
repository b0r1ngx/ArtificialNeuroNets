from neurolab import net as n, trans
from Lab00 import generation_n_visualisation
import matplotlib.pyplot as plt
from numpy import random as r, array

# Dataset 1. Exercises 1, 2, 3, 4.
P, T = generation_n_visualisation.noughts_n_crosses()

net = n.newff(
    [[0, 1], [0, 1]],
    [6, 5, 1],
    [trans.HardLim()] * 3
)
print(f"""Inputs: {net.ci},
Outputs: {net.co},
Layers: {len(net.layers)},
Connection scheme of layers: {net.connect}""")

# Change weight of input (first) layer
net.layers[0].np['w'] = [[1, 0],
                         [1, 0],
                         [1, 0],
                         [0, 1],
                         [0, 1],
                         [0, 1]]
# Change bias of input (first) layer
net.layers[0].np['b'] = [-.25, -.5, -.75, -.25, -.5, -.75]

# Change weight of second layer
net.layers[1].np['w'] = [[0, 0, 1, 0, 0, 1],
                         [0, 0, 1, 1, -1, 0],
                         [1, 0, -1, -1, 0, 0],
                         [1, -1, 0, 0, 0, -1],
                         [0, 0, -1, 0, 1, -1]]
# Change bias of second layer
net.layers[1].np['b'] = [-1.5, -1.5, -.5, -.5, -.5]

# Change weight of output (third) layer
net.layers[2].np['w'] = [[1, 1, 1, 1, 1]]
# Change bias of output (third) layer
net.layers[2].np['b'] = [-0.5]

rnd_input = r.rand(10000, 2)
sim = net.sim(rnd_input)


def show_sim(dots, sim, title: str):
    crosses = []
    noughts = []

    for it in range(len(sim)):
        if sim[it] == 1:
            crosses.append(dots[it])
        elif sim[it] == 0:
            noughts.append(dots[it])

    crosses = array(crosses)
    noughts = array(noughts)

    if len(crosses) != 0:
        plt.plot(crosses[:, 0], crosses[:, 1], 'r.')
    if len(noughts) != 0:
        plt.plot(noughts[:, 0], noughts[:, 1], 'b.')

    plt.title(title)
    plt.show()


show_sim(rnd_input, sim, "")

# Exercise 5.
#   Try to solve the recognition problem by training 1-layer
#   perceptron on the set of input examples. Analyze the result.

# prepare data
train_size = int(len(rnd_input) * 0.75)

rnd_input_train, rnd_input_test = rnd_input[:train_size], rnd_input[train_size:]
sim_train, sim_test = sim[:train_size], sim[train_size:]

# net = n.newp(
#     [[0, 1], [0, 1]],
#     1
# )
net = n.newff(
    [[0, 1], [0, 1]],  # list of list the outer list is the number of input neurons, inner is min max for this input
    [6, 5, 1]  # contains the number of neurons for each layer, size of it number of layers, except input layer
)
net.init()
print(f"""Train function: {net.trainf},
Error function w/ derivative: {net.errorf}""")

print('start train net')
error = net.train(rnd_input_train, sim_train, epochs=20, show=1, goal=0.001, rr=0.01)
print('Error, after train: ', error)

out_train = net.sim(rnd_input_test)
print(out_train)

# Self-added exercise 6.*
#   Try to solve the recognition problem by training neuro-net
#   on the set of input examples. Analyze the result.

print('end')
