# coding: utf-8
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[1, 0],
              [0, 1],
              [0, 1],
              [1, 0]])


def draw_decision_boundary(bp, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))
    Z = bp.predict(np.c_[xx.ravel(), yy.ravel()]).T
    Z = Z.argmax(axis=1)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


class Backpropagation(object):

    def __init__(self, layers, alpha=0.3, activation=sigmoid):
        self.activation = activation
        self.alpha = alpha
        self.weights = []
        self.biases = []
        self.layers = layers
        for i, n in enumerate(layers):
            if i != len(layers) - 1:
                self.weights.append(np.random.rand(layers[i + 1], n))
                self.biases.append(np.random.rand(layers[i + 1], 1))

    def _forward_pass(self, X):
        self.layer_activations = [X.T]
        for i, weight in enumerate(self.weights):
            activation = np.dot(weight, self.layer_activations[i]) + \
                    self.biases[i]
            self.layer_activations.append(self.activation(activation))
        return self.layer_activations[-1]

    def predict(self, X):
        return self._forward_pass(X)

#    def predict(self, X):
#        if X.ndim == 1:
#            X = X.reshape(-1, 1)
#        else:
#            X = X.T
#        self.a1 = X
#        z2 = np.dot(self.weights[0], self.a1) + self.biases[0]
#        self.a2 = sigmoid(z2)
#        z3 = np.dot(self.weights[1], self.a2) + self.biases[1]
#        self.a3 = sigmoid(z3)
#        return self.a3

    def fit(self, X, y, n_iter=1000, showloss=True):
        if showloss:
            self.losses = []
        for i in range(n_iter):
            output = self._forward_pass(X)
            if showloss:
                loss = ((output - y.T) ** 2).sum()
                print loss
                self.losses.append(loss)

            # compute local gradients
            local_deltas = []
            local_gradients = []
            # last layer
            del_op = - (y.T - output) * output * (1 - output)
            local_deltas.append(del_op)
            grad_op = np.dot(del_op, self.layer_activations[-2].T), \
                del_op.sum(1).reshape(-1, 1)
            local_gradients.append(grad_op)

            for layer_ix in range(len(self.layers) - 2, -1, -1):
                if layer_ix != 0:
                    local_activation = self.layer_activations[layer_ix]
                    delta = np.dot(self.weights[layer_ix].T,
                                local_deltas[-1]) * \
                        local_activation * (1 - local_activation)
                    grad_w = np.dot(local_deltas[-1],
                                    local_activation.T).T
                    grad_b = delta.sum(1).reshape(-1, 1)
                    local_deltas.append(delta)
                    local_gradients.append((grad_w, grad_b))
            local_gradients.reverse()
            for i in range(len(local_gradients)):
                gradw, gradb = local_gradients[i]
                self.weights[i] -= self.alpha * gradw
                self.biases[i] -= self.alpha * gradb


#            gradw2 = np.dot(del3, self.a2.T)
#            gradb2 = del3.sum(1).reshape(-1, 1)
#            gradw1 = np.dot(del2, self.a1.T)
#            gradb1 = del2.sum(1).reshape(-1, 1)
#
#            self.w1 -= self.alpha * gradw1
#            self.w2 -= self.alpha * gradw2
#            self.b1 -= self.alpha * gradb1
#            self.b2 -= self.alpha * gradb2

if __name__ == '__main__':
    X, y = make_circles(noise=0.05, factor=0.3)
    bp = Backpropagation(layers=[2, 3, 2])
    bp.fit(X, y, n_iter=100000)
    draw_decision_boundary(bp, X, y)
