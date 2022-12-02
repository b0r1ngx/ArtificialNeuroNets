import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import *

from _2022.AR2model import generate_model_data_with_plt
from _2022.lab01.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork
from _2022.utils import prepare_data, train_test_split

# Report - https://docs.google.com/document/d/13QCUBmh3GJXVHs6OscQK25FowSgfTEXit2p0OctxYs8

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Used device: {device}")
torch.device(device)

input_size = 10

feed_forward_nn = FeedForwardNeuralNetwork(input_layer=input_size)
print(feed_forward_nn)

window_sizes = (1, 10, 50, 100)
delays = (0, 10, 100, 1000)

# Generate Data
data = generate_model_data_with_plt(n=20000)
# other_test_data = ar2_model_with_plt(n=20000)
print("Size of all data from ar2_model:", len(data))
current_data = prepare_data(data, window_size=input_size)
# other_test_data = split_data(other_test_data, window_size=input_size, delay=10)

size = len(current_data)
batch_size = len(current_data[0][0])
print("Size after split_data:", size)
print("Batch size:", batch_size)

train_data, test_data = train_test_split(current_data, train=0.5)
print("Sizes of data, after train_test_split:", len(train_data), len(test_data))

loss_function = MSELoss()
learning_rate = 5e-6
optimizer = torch.optim.Adamax(feed_forward_nn.parameters(), lr=learning_rate)


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


prev_loss = 0


def test(data: list[list[Tensor]], model, loss_fn):
    size = len(data)
    num_batches = size / batch_size
    model.eval()
    test_loss, correct = 0, 0
    corrects, predictions = [], []
    with torch.no_grad():
        for d in data:
            y, y_pred = d
            pred = model(y)
            test_loss += loss_fn(pred, y_pred).item()

            corrects.append(y_pred)
            predictions.append(pred)

    test_loss /= num_batches
    print(f"Global loss (for {size}): {test_loss:>8f}")
    print(f"Local loss: {test_loss / size:>8f}")

    plt.plot(corrects, 'g')
    plt.plot(predictions, 'r')
    plt.legend(['correct', 'prediction'])
    plt.show()
    global prev_loss
    print(f"Avg loss win: {prev_loss - test_loss:>8f} \n")
    prev_loss = test_loss


epochs = 1000
for t in range(epochs):
    epoch = t + 1
    print(f"Epoch {epoch}\n-------------------------------")
    plt.title(f'2.{epoch} test difference')
    train(train_data, feed_forward_nn, loss_function, optimizer)
    test(test_data, feed_forward_nn, loss_function)

print("Done!")

# The last linear layer of the neural network returns logits -
# raw values in [-infty, infty] - which are passed to the nn.Softmax module.
# The logits are scaled to values [0, 1] representing the model’s predicted
# probabilities for each class. dim parameter indicates the dimension
# along which the values must sum to 1.
# pred_probab = softmax(out)
