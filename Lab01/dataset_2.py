from neurolab import net, trans
from Lab00 import generation_n_visualisation
from numpy import random as r

# Dataset 2. Exercises 1, 2, 3, 4.
P, T = generation_n_visualisation.boolean_function()

net = net.newff([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]], [7, 1], [trans.HardLim()] * 2)
print(f"""Inputs: {net.ci},
Outputs: {net.co},
Layers: {len(net.layers)},
Connection scheme of layers: {net.connect}""")

net.layers[0].np['w'] = [[-1, -1, -1, -1, -1],
                         [-1, -1, -1, -1, 1],
                         [-1, -1, -1, 1, -1],
                         [-1, -1, 1, -1, -1],
                         [-1, 1, -1, -1, -1],
                         [1, -1, -1, -1, -1],
                         [1, 1, 1, 1, 1]]
net.layers[0].np['b'] = [.5, -.5, -.5, -.5, -.5, -.5, -4.5]

net.layers[1].np['w'] = [[1, 1, 1, 1, 1, 1, 1]]
net.layers[1].np['b'] = [-.5]

sim = net.sim(P)
print("Real answer:", T)

print("Simulated answer:", sim[:32])

# Exercise 5.
#   Try to train a single layer perceptron on your function using
#   fragments of the truth table as a training sample. Analyze
#   results and calculate the mean error.
net = net.newp([[0, 1], [0, 1]], 1)
net.init()