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

alpha = 0.3

for i in range(100):

    for j in range(X.shape[0]):
        sample = X[j, :]
        target = y[j, :]

        a1 = sample.reshape(-1, 1)
        z2 = np.dot(w1, a1) + b1
        a2 = sigmoid(z2)
        z3 = np.dot(w2, a2) + b2
        a3 = sigmoid(z3)

        loss = ((a3 - target) ** 2).sum()
        print loss
        losses.append(loss)

        target = target.reshape(-1, 1)
        del3 = - (target - a3) * a3 * (1 - a3)
        del2 = np.dot(w2.T, del3) * a2 * (1 - a2)
        gradw2 = np.dot(del3, a2.T)
        gradb2 = del3
        gradw1 = np.dot(del2, a1.T)
        gradb1 = del2

        w1 -= alpha * gradw1
        w2 -= alpha * gradw2
        b1 -= alpha * gradb1
        b2 -= alpha * gradb2

plt.plot(losses)
plt.show()
