# coding: utf-8
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np


layers = [64, 128, 10]


def get_data():
    digits = load_digits()
    digits.keys()
    X = digits['data']
    y = digits['target']
    y = LabelBinarizer().fit_transform(y)
    X = StandardScaler().fit_transform(X)
    return X, y


def sigmoid(x, derivative=False):
    if not derivative:
        return 1.0 / (1 + np.exp(-x))
    out = sigmoid(x)
    return out * (1 - out)


class Backpropagation(object):

    def __init__(self, layers=layers, alpha=0.3):
        self.layers = layers
        self.layerCount = len(self.layers) - 1
        self.alpha = alpha
        self.weights = []

        # I/O data from the last run
        self._layerInput = []
        self._layerOutput = []

        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            weight = np.random.rand(l2, l1 + 1)
            self.weights.append(weight)

    def trainEpoch(self, X, y):
        y = np.array(y)
        delta = []
        lnCases = X.shape[0]

        # first run the network
        self.predict(X)

        # Calculate error deltas
        for i in range(self.layerCount)[::-1]:
            if i == self.layerCount - 1:
                # compare output with target values
                output_delta = self._layerOutput[i] - y.T
                error = np.sum(output_delta ** 2)
                _delta = output_delta * sigmoid(self._layerInput[i], True)
                delta.append(_delta)
            else:
                # compare to the following layer's delta
                w = self.weights[i + 1].T
                delta_pullback = np.dot(w, delta[-1])
                delta.append(delta_pullback[:-1, :] *
                        sigmoid(self._layerInput[i], True))
        # Computing the weightdeltas
        for i in range(self.layerCount):
            delta_index = self.layerCount - 1 - i
            if i == 0:
                layerOutput = np.c_[X, np.ones((lnCases, 1))].T
            else:
                layerOutput = np.r_[self._layerOutput[i - 1],
                                    np.ones((1,
                                        self._layerOutput[i - 1].shape[1]))]
            weightDelta = np.sum(layerOutput[None, :, :].transpose(2, 0, 1) *
                    delta[delta_index][None, :, :].transpose(2, 1, 0),
                    axis=0)
            self.weights[i] -= self.alpha * weightDelta
        return error

    def predict(self, data):
        if data.ndim == 1:
            data = data.reshape(1, -1)
        lnCases = data.shape[0]
        self._layerInput = []
        self._layerOutput = []

        # running it
        for i in range(self.layerCount):
            if i == 0:
                data_biased = np.c_[data, np.ones((data.shape[0], 1))]
                layerInput = np.dot(self.weights[i], data_biased.T)
            else:
                data_biased = np.c_[self._layerOutput[-1].T,
                                    np.ones((lnCases, 1))]
                layerInput = np.dot(self.weights[i], data_biased.T)
            self._layerInput.append(layerInput)
            self._layerOutput.append(sigmoid(layerInput))
        return self._layerOutput[-1].T
