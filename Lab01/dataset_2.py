from neurolab import net, trans
from Lab00 import generation_n_visualisation
from numpy import random as r, asfarray

# Dataset 2
P, T = generation_n_visualisation.boolean_function()

# transf = trans.HardLim()
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
print(sim)
