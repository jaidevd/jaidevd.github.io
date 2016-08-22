# coding: utf-8
import numpy as np
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

layers = [2, 3, 2]

w1 = np.random.rand(3, 2)
w2 = np.random.rand(2, 3)
b1 = np.random.rand(3, 1)
b2 = np.random.rand(2, 1)
losses = []

alpha = 0.5


def run(w1, w2, b1, b2):
    for i in range(1000000):

        a1 = X.T
        z2 = np.dot(w1, a1) + b1
        a2 = sigmoid(z2)
        z3 = np.dot(w2, a2) + b2
        a3 = sigmoid(z3)

        loss = ((a3 - y.T) ** 2).sum()
        print loss
        losses.append(loss)

        del3 = - (y.T - a3) * a3 * (1 - a3)
        del2 = np.dot(w2.T, del3) * a2 * (1 - a2)
        gradw2 = np.dot(del3, a2.T)
        gradb2 = del3.sum(1).reshape(-1, 1)
        gradw1 = np.dot(del2, a1.T)
        gradb1 = del2.sum(1).reshape(-1, 1)

        w1 -= alpha * gradw1
        w2 -= alpha * gradw2
        b1 -= alpha * gradb1
        b2 -= alpha * gradb2

    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    run(w1, w2, b1, b2)
