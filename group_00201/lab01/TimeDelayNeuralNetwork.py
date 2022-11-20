from torch import nn


class TimeDelayNeuralNetwork(nn.Module):
    def __init__(self,
                 input_layer=10,
                 neurons_first_layer=20,
                 neurons_second_layer=20,
                 output_layer=1):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(input_layer, neurons_first_layer),
            nn.ReLU(),  # nn.Threshold(0.5, 0)
            nn.Linear(neurons_first_layer, neurons_second_layer),
            nn.ReLU(),  # nn.Threshold(0.5, 0)
            nn.Linear(neurons_second_layer, output_layer)
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        logits = self.architecture(x)
        return logits