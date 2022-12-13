import numpy as np
from matplotlib import cm, pyplot as plt


class HopfieldNetwork:
    """https://github.com/takyamamoto/Hopfield-Network"""

    def train_weights(self, train_data):
        print("Start to train weights...")
        train_data = np.array(train_data)
        num_data = len(train_data)
        self.num_neuron = train_data[0].shape[0]

        # initialize weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neuron)

        # Hebb rule
        for i in range(num_data):
            t = train_data[i] - rho
            W += np.outer(t, t)

        # Make diagonal element of W into 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data

        self.W = W

    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        data = np.array(data)

        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn

        # Copy to avoid call by reference
        copied_data = np.copy(data)

        # Define predict list
        predicted = []
        for i in range(len(data)):
            predicted.append(self._run(copied_data[i]))
        return predicted

    def _run(self, init_s):
        if self.asyn:
            # Compute initial state energy
            s = init_s
            e = self.energy(s)

            for i in range(self.num_iter):
                for j in range(100):
                    # Select random neuron
                    idx = np.random.randint(0, self.num_neuron)
                    # Update s
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)

                # Compute new state energy
                e_new = self.energy(s)
                # s is converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new

            return s
        else:
            s = init_s
            e = self.energy(s)

            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)

                e_new = self.energy(s)
                if e == e_new:
                    return s

                e = e_new

            return s

    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network Weights")
        plt.tight_layout()
        plt.show()
