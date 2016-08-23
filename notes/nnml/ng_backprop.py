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

    def __init__(self, alpha=0.3, activation=sigmoid):
        self.activation = activation
        self.alpha = alpha

    def predict(self, X):
        for attr in "w1 w2 b1 b2".split():
            if not hasattr(self, attr):
                raise ValueError("Please call fit first.")
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        else:
            X = X.T
        self.a1 = X
        z2 = np.dot(self.w1, self.a1) + self.b1
        self.a2 = sigmoid(z2)
        z3 = np.dot(self.w2, self.a2) + self.b2
        self.a3 = sigmoid(z3)
        return self.a3

    def fit(self, X, y, n_iter=1000, showloss=True):
        if showloss:
            self.losses = []
        self.w1 = np.random.rand(3, 2)
        self.w2 = np.random.rand(2, 3)
        self.b1 = np.random.rand(3, 1)
        self.b2 = np.random.rand(2, 1)
        for i in range(n_iter):
            self.predict(X)
            if showloss:
                loss = ((self.a3 - y.T) ** 2).sum()
                print loss
                self.losses.append(loss)
            del3 = - (y.T - self.a3) * self.a3 * (1 - self.a3)
            del2 = np.dot(self.w2.T, del3) * self.a2 * (1 - self.a2)
            gradw2 = np.dot(del3, self.a2.T)
            gradb2 = del3.sum(1).reshape(-1, 1)
            gradw1 = np.dot(del2, self.a1.T)
            gradb1 = del2.sum(1).reshape(-1, 1)

            self.w1 -= self.alpha * gradw1
            self.w2 -= self.alpha * gradw2
            self.b1 -= self.alpha * gradb1
            self.b2 -= self.alpha * gradb2

if __name__ == '__main__':
    X, y = make_circles(noise=0.05, factor=0.3)
    bp = Backpropagation()
    bp.fit(X, y, n_iter=100000)
    draw_decision_boundary(bp, X, y)
