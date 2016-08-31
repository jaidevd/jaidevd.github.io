# coding: utf-8
import numpy as np
from tqdm import tqdm
# from scipy.io import savemat
from sklearn.datasets import make_blobs
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


np.random.seed(42)
STD = 0.35


# layer1 = []
# layer2 = []
# layer3 = []


def _get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)


def make_problem():
    pos_center = np.array([[-1, -1], [1, 1]])
    pos_data = make_blobs(centers=pos_center, cluster_std=STD)[0]
    pos_data = np.c_[pos_data, np.ones((pos_data.shape[0], 1))]

    neg_center = np.array([[1, -1], [-1, 1]])
    neg_data = make_blobs(centers=neg_center, cluster_std=STD)[0]
    neg_data = np.c_[neg_data, np.zeros((neg_data.shape[0], 1))]

    X = np.r_[pos_data, neg_data]
    np.random.shuffle(X)
    y = OneHotEncoder().fit_transform(X[:, 2].reshape(-1, 1))
    X = X[:, :2]
    return X, y.toarray()


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def draw_decision_boundary(bp, X, y):
    y = np.argmax(y, axis=1)
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
                self.weights.append(_get_weights((layers[i + 1], n)))
                self.biases.append(_get_weights((layers[i + 1], 1)))

    def _forward_pass(self, X):
        self.layer_activations = [X.T]
        for i, weight in enumerate(self.weights):
            activation = np.dot(weight, self.layer_activations[i]) + \
                self.biases[i]
            self.layer_activations.append(self.activation(activation))
        return self.layer_activations[-1]

    def predict(self, X):
        return self._forward_pass(X)

    def fit(self, X, y, n_iter=1000, showloss=True):
        if showloss:
            self.losses = []
        for _ in tqdm(xrange(n_iter)):
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
                local_activation = self.layer_activations[layer_ix]
                delta = np.dot(self.weights[layer_ix].T,
                            local_deltas[-1]) * \
                    local_activation * (1 - local_activation)
                local_deltas.append(delta)
                grad_w = np.dot(local_deltas[-1],
                                self.layer_activations[layer_ix - 1].T)
                grad_b = delta.sum(1).reshape(-1, 1)
                local_gradients.append((grad_w, grad_b))
            local_gradients.reverse()
            for i, (gradw, gradb) in enumerate(local_gradients[1:]):
                self.weights[i] -= self.alpha * gradw
                self.biases[i] -= self.alpha * gradb
#            layer1.append(np.r_[self.weights[0].ravel(), self.biases[0].ravel()])
#            layer2.append(np.r_[self.weights[1].ravel(), self.biases[1].ravel()])
#            layer3.append(np.r_[self.weights[2].ravel(), self.biases[2].ravel()])
#        savemat('generalized_backprop.mat', dict(layer1=np.array(layer1),
#            layer2=np.array(layer2), layer3=np.array(layer3)))


if __name__ == '__main__':
    X, y = make_problem()
    X = StandardScaler().fit_transform(X)
    bp = Backpropagation(layers=[2, 3, 2, 2])
    bp.fit(X, y, n_iter=1000000)
    draw_decision_boundary(bp, X, y)
    plt.plot(bp.losses)
    plt.show()
