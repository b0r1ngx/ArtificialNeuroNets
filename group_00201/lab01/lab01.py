import random as r
from neurolab import net as n, trans


def ar_model(x1, x2, a1, a2, c=0, eps=0.0):
    return c + a1 * x1 + a2 * x2 + eps


temperatures_x1 = [7, 8, 10, 9, 10,
                   10, 11, 12, 14, 11,
                   12, 13, 12, 13, 14,
                   15, 20, 16, 18, 20,
                   22, 21, 23, 22, 23,
                   25, 27, 28, 29, 30, 28]

reversed_temperatures = temperatures_x1[::-1]

UV_x2 = [1, 2, 3, 3, 3,
         4, 3, 5, 5, 2,
         3, 4, 4, 3, 4,
         3, 5, 7, 8, 5,
         4, 6, 4, 6, 5,
         5, 7, 8, 9, 9, 7]

reversed_UV = UV_x2[::-1]

a1 = 1
a2 = 1

c = 1

prepared_temperatures_x1 = []
prepared_UV_x2 = []
for i in range(100):
    prepared_temperatures_x1 += temperatures_x1
    prepared_temperatures_x1 += reversed_temperatures

    prepared_UV_x2 += UV_x2
    prepared_UV_x2 += reversed_UV

data_size = len(prepared_temperatures_x1)
print(data_size)

y = []
for i in range(data_size):
    y.append(
        ar_model(
            prepared_temperatures_x1[i],
            prepared_UV_x2[i],
            a1,
            a2,
            c,
            r.random()
        )
    )

train = int(data_size * 0.75)

input_train, input_test = [], []
output_train, output_test = [], []
for i in range(train):
    input_train.append([prepared_temperatures_x1[i], prepared_UV_x2[i]])
    output_train.append([y[i]])

for i in range(train, data_size):
    input_test.append([prepared_temperatures_x1[i], prepared_UV_x2[i]])
    output_test.append([y[i]])

net = n.newff(
    [[0, 50], [0, 10]],  # list of list the outer list is the number of input neurons, inner is min max for this input
    [10, 10, 10, 1]  # contains the number of neurons for each layer, size of it number of layers, except input layer
)

# is that right, reset trans function
net = n.newff(
    [[0, 50], [0, 10]],  # list of list the outer list is the number of input neurons
    [20, 1],  # contains the number of neurons for each layer, size of it number of layers, except input layer
    [trans.HardLim()] * 2  # list of activation function for each layer
)
print(f"""Inputs: {net.ci},
Outputs: {net.co},
Layers: {len(net.layers)},
Connection scheme of layers: {net.connect}""")

# what is input to input data
error = net.train(input_train, output_train, epochs=50, show=10, goal=0.0001)
print(error)
out_train = net.sim(input_train)

print(out_train)

out_test = net.sim(input_test)
print(out_train)

# second chance, try to implement this on practice lesson, yet not undrstand
# is that right, reset trans function
net = n.newff(
    [[7, 30], [1, 9]],  # list of list the outer list is the number of input neurons, inner is min max for this input
    [10, 1]  # contains the number of neurons for each layer, size of it number of layers, except input layer
)
net.init()
print(f"""Inputs: {net.ci},
Outputs: {net.co},
Layers (include output): {len(net.layers)},
Connection scheme of layers: {net.connect}""")

# what is input to input data
error = net.train(input_train, output_train, epochs=16000, show=1000, goal=0.00001)
print('Error, after train: ', error)
out_train = net.sim(input_test)
print('', out_train)

# net.init()
print(f"""Inputs: {net.ci},
Outputs: {net.co},
Layers (include output): {len(net.layers)},
Connection scheme of layers: {net.connect}""")

print(f"""Train function: {net.trainf},
Error function w/ derivative: {net.errorf}""")

# what is input to input data
error = net.train(input_train, output_train, epochs=100, show=1, goal=1, rr=0.1)
print('Error, after train: ', error)

out_test = net.sim(input_test)
print('out', out_test)

# Verdict: I can't work with neurolab (Lib),
# its totally not so good, or I'm so bad
