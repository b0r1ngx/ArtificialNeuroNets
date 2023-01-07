import random

import numpy as np
from matplotlib import cm, pyplot as plt


class HopfieldNetwork:
    """Source: https://github.com/takyamamoto/Hopfield-Network,
        but changed a bit"""

    def train_weights(self, train_data):
        print("Start to train weights...")
        train_data = np.array(train_data)
        data_size = len(train_data)
        self.neurons = train_data[0].shape[0]

        # initialize weights
        weights = np.zeros((self.neurons, self.neurons))
        rho = np.sum([np.sum(t) for t in train_data]) / (data_size * self.neurons)

        # Hebb rule
        for i in range(data_size):
            t = train_data[i] - rho
            weights += np.outer(t, t)

        # Make diagonal element of weights - 0,
        # and finish on weights
        diagonal_weights = np.diag(np.diag(weights))
        weights = weights - diagonal_weights
        weights /= data_size

        self.weights = weights

    def predict(self, data, iterations_limit=20, bias=0, sync=True):
        data = np.array(data)

        self.iterations_limit = iterations_limit
        self.bias = bias
        self.sync = sync

        # Copy to avoid call by reference
        copied_data = np.copy(data)

        # Define predict list
        predicted = []
        for i in range(len(data)):
            predicted.append(
                self.find_balance_cycles(
                    copied_data[i]
                )
            )
        return predicted

    def find_balance_cycles(self, start_state):
        if self.sync:
            # Compute initial state energy
            state = start_state
            energy = self.energy(state)

            for i in range(self.iterations_limit):
                # Update state
                state = np.sign(self.weights @ state - self.bias)

                # Compute new state energy
                new_energy = self.energy(state)
                if energy == new_energy:
                    print(f'NN(sync) - stable state reached in {i + 1} cycles')
                    return state

                energy = new_energy

            print('NN(sync) - stable state not reached')
            return state
        else:
            state = start_state
            energy = self.energy(state)

            for i in range(self.iterations_limit):
                indexes = list(range(self.neurons))
                for j in range(self.neurons):
                    # Select random neuron idx = np.random.randint(0, self.neurons)
                    # Instead of just visit every time a random neurons, we visit it all, but still in random order
                    idx = random.choice(indexes)
                    indexes.remove(idx)
                    # Update state
                    state[idx] = np.sign(self.weights[idx].T @ state - self.bias)

                new_energy = self.energy(state)
                if energy == new_energy:
                    print(f'NN(async) - stable state reached in {i + 1} cycles')
                    return state

                energy = new_energy

            print('NN(async) - stable state not reached')
            return state

    def energy(self, state):
        return -.5 * state @ self.weights @ state + np.sum(state * self.bias)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.weights, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title(f'Network weights (neurons = {self.neurons})')
        plt.tight_layout()
        plt.show()
