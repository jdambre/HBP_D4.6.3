import numpy as np


class FastNoisyReservoir(object):
    def __init__(self, input_dims, num_nodes, input_scaling=0.1, spec_rad=0.5):
        self.input_dims = input_dims
        self.num_nodes = num_nodes
        self.W = np.random.randn(self.num_nodes, 1 + self.input_dims + self.num_nodes) * 0.05
        self.W[:, self.input_dims + 1:] = self.W[:, self.input_dims + 1:] / np.max(
            np.abs(np.linalg.eigvals(self.W[:, self.input_dims + 1:]))) * spec_rad
        self.W_hist = []
        self.state = np.zeros(self.num_nodes)

    def copy(self):
        r = FastNoisyReservoir(self.input_dims, self.num_nodes)
        r.W = self.W.copy()
        r.W_hist = self.W_hist
        r.state = self.state.copy()
        return r

    def run(self, U, noise=0.01, mask=1., cov=None):
        X_store = []
        Z_store = []  # noise
        Y_store = []
        state_store = []  # states with noise

        for i in xrange(U.shape[0]):
            X = np.hstack((np.ones((1, 1)), U[i:i + 1, :], self.state.reshape((1, self.state.shape[0])))).reshape(
                (-1, 1))
            Y = np.dot(self.W, X)  # N x 1: neuron activations without noise
            if (cov is None):
                Z = np.random.randn(self.num_nodes) * noise * mask
            else:
                pass  # np.random.mutivariate_normal(np.zeros(self.num_nodes),)
            self.state = np.tanh(Y.ravel() + Z)
            # Y_store.append(Y.ravel())
            state_store.append(self.state.copy())
            X_store.append(X.ravel())
            Z_store.append(Z)
        return np.array(X_store), np.array(Z_store), np.array(Y_store), np.array(state_store)

    def learn(self, reward, reward_mean, Z, X, alpha, norm=None, mask=None):
        e = np.dot(Z.T, X)

        # normalization (doesn't seem to work in RNN), only in feedforward NN
        # similar to Oja's rule
        if (not norm is None):
            wes = (e * self.W).sum(1) / (norm ** 2)
            e -= (wes * self.W.T).T

        #equivalent:
        #e-=np.dot(np.diag(np.diag(np.dot(e,self.W.T))),self.W)/norm**2

        deltaW = (alpha * (reward - reward_mean) * e)
        if (not mask is None):
            deltaW *= mask
        self.W += deltaW
        return deltaW

