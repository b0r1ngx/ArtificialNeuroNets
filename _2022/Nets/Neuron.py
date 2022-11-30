class Neuron(w, x):
    pre_activation_output = 0

    def pre_activation(self, x, w):
        return w * x

    def activation(self):
        if self.pre_activation_output >= 0:
            return 1
        return 0
