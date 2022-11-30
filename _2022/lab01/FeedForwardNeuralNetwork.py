from torch import nn


class FeedForwardNeuralNetwork(nn.Module):
    def __init__(self, input_layer=10, neurons_first_layer=10, output_layer=1):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_layer, neurons_first_layer),
            nn.Threshold(0.5, 0),
            nn.Linear(neurons_first_layer, neurons_first_layer),
            nn.Threshold(0.5, 0),
            nn.Linear(neurons_first_layer, output_layer)
        )

    def forward(self, x):
        return self.architecture(x)
