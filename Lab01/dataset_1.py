from neurolab import net as n, tool, trans
from Lab00 import generation_n_visualisation
from numpy import random as r

# Dataset 1. Exercises 1, 2, 3, 4
P, T = generation_n_visualisation.noughts_n_crosses()
# minmax - [0, 0,75]
net = n.newff([[0, 1], [0, 1]], [6, 5, 1], [trans.HardLim()] * 3)

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
# [1], [1], [1], [1], [1]
# Change bias of output (third) layer
net.layers[2].np['b'] = [-0.5]

print(net.layers[0].np)
print(net.layers[1].np)
print(net.layers[2].np)

rnd_input = r.rand(10000, 2)
sim = net.sim(rnd_input)
print(rnd_input)
print(sim)


# Exercise 5
net = n.newp([[0, 1], [0, 1]], 1)
net.init()

error = net.train(P, T, epochs=1000, )
print(error)
print(net.trainf.__doc__)

# print(f"""Train function: {net.trainf},
# Error function w/ derivative: {net.errorf}""")
