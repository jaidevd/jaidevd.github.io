# coding: utf-8
import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

sample = np.array([[0, 0]])
target = np.array([[1, 0]])

layers = [2, 3, 2]

w1 = np.random.rand(3, 2)
w2 = np.random.rand(2, 3)
b1 = np.random.rand(3, 1)
b2 = np.random.rand(2, 1)

a1 = sample.T
z2 = np.dot(w1, a1) + b1
a2 = sigmoid(z2)
z3 = np.dot(w2, a2) + b2
a3 = sigmoid(z3)
target = target.T
del3 = - (target - a3) * a3 * (1 - a3)
del2 = np.dot(w2.T, del3) * a2 * (1 - a2)
gradw2 = np.dot(del3, a2.T)
gradb2 = del3
gradw1 = np.dot(del2, a1.T)
gradb1 = del2
