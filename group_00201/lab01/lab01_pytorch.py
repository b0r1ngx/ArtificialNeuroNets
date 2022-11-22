import torch
from matplotlib import pyplot as plt
from torch import Tensor, tensor
from torch.nn import *

from group_00201.AR2model import ar2_model_with_plt
from group_00201.lab01.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
from group_00201.utils import split_data, train_test_split

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Used device: {device}")
torch.device(device)

input_size = 10

time_delay_nn = FeedForwardNeuralNetwork(input_layer=input_size)
print(time_delay_nn)

window_sizes = (1, 10, 50, 100)
delays = (0, 10, 100, 1000)

# Generate Data
data = ar2_model_with_plt(n=15000)
print("Size of all data from ar2_model:", len(data))
current_data = split_data(data, window_size=input_size)

size = len(current_data)
batch_size = len(current_data[0][0])
print("Size after split_data:", size)
print("Batch size:", batch_size)

train_data, test_data = train_test_split(current_data, train=0.5)
print("Sizes of data, after train_test_split:", len(train_data), len(test_data))

loss_function = MSELoss()
learning_rate = 1e-6
optimizer = torch.optim.SGD(time_delay_nn.parameters(), lr=learning_rate)

prev_loss = 0


def train(data: list[list[Tensor]], model, loss_function, optimizer):
    size = len(data)
    num_batch = 0
    model.train()
    for d in data:
        y, y_pred = d

        # Compute prediction error
        pred = model(y)
        loss = loss_function(pred, y_pred)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_batch += 1
        if num_batch % 1000 == 0:
            loss, current = loss.item(), num_batch
            print(f"loss: {loss:>7f} [{current:>5d} /{size:>5d}]")


def test(data: list[list[Tensor]], model, loss_fn):
    size = len(data)
    num_batches = size / batch_size
    model.eval()
    test_loss, correct = 0, 0
    corrs = []
    preds = []
    with torch.no_grad():
        for d in data:
            y, y_pred = d
            pred = model(y)
            test_loss += loss_fn(pred, y_pred).item()

            corrs.append(y_pred)
            preds.append(pred)

    test_loss /= num_batches
    print(f"Global loss (for {size}): {test_loss:>8f}")
    print(f"Local loss: {test_loss / size:>8f}")

    plt.plot(corrs, 'g')
    plt.plot(preds, 'r')
    plt.legend(['correct', 'prediction'])
    plt.show()
    global prev_loss
    print(f"Avg loss win: {prev_loss - test_loss:>8f} \n")
    prev_loss = test_loss


epochs = 100
for t in range(epochs):
    _t = t + 1
    print(f"Epoch {_t}\n-------------------------------")
    plt.title(f'2.{_t} test difference')
    train(train_data, time_delay_nn, loss_function, optimizer)
    test(test_data, time_delay_nn, loss_function)

print("Done!")

# The last linear layer of the neural network returns logits -
# raw values in [-infty, infty] - which are passed to the nn.Softmax module.
# The logits are scaled to values [0, 1] representing the model’s predicted
# probabilities for each class. dim parameter indicates the dimension
# along which the values must sum to 1.
# pred_probab = softmax(out)
