import torch
from matplotlib import pyplot as plt
from torch import Tensor, tensor
from torch.nn import *

from group_00201.AR2model import ar2_model_with_plt
from group_00201.lab01.TimeDelayNeuralNetwork import TimeDelayNeuralNetwork
from group_00201.utils import split_data, train_test_split

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Used device: {device}")
torch.device(device)

time_delay_nn = TimeDelayNeuralNetwork()
print(time_delay_nn)

window_sizes = (1, 10, 50, 100)
delays = (0, 10, 100, 1000)

# Generate Data
data = ar2_model_with_plt()
print("Size of all data from ar2_model:", len(data))
current_data = split_data(data)

size = len(current_data)
batch_size = len(current_data[0][0])
print("Size after split_data:", size)
print("Batch size:", batch_size)

train_data, test_data = train_test_split(current_data)
print("Sizes of data, after train_test_split:", len(train_data), len(test_data))

loss_function = MSELoss()  # CrossEntropyLoss()
learning_rate = 1e-5
optimizer = torch.optim.SGD(time_delay_nn.parameters(), lr=learning_rate)

prev_loss = 0


def first_method():
    def train(data: list[list[Tensor]], model, loss_function, optimizer):
        size = len(data)
        num_batch = 0
        # model.train()
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
        # model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for d in data:
                y, y_pred = d
                pred = model(y)
                test_loss += loss_fn(pred, y_pred).item()
                correct += (pred.argmax(0) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        global prev_loss
        print(f"Avg loss win: {test_loss - prev_loss:>8f} \n")
        prev_loss = test_loss

    epochs = 100
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_data, time_delay_nn, loss_function, optimizer)
        test(test_data, time_delay_nn, loss_function)
    print("Done!")


def second_method(model):
    y, y_pred = [], []
    for _ in train_data:
        y.append(train_data[0])
        y_pred.append(train_data[1])

    y, y_pred = tensor(y), tensor(y_pred)
    losses = []
    for epoch in range(5000):
        pred_y = model(y)
        loss = loss_function(pred_y, y_pred)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f" % (learning_rate))
    plt.show()


first_method()
# The last linear layer of the neural network returns logits -
# raw values in [-infty, infty] - which are passed to the nn.Softmax module.
# The logits are scaled to values [0, 1] representing the model’s predicted
# probabilities for each class. dim parameter indicates the dimension
# along which the values must sum to 1.
# pred_probab = softmax(out)
