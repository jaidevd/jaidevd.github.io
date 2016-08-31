#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Cube26 product code
#
# (C) Copyright 2015 Cube26 Software Pvt Ltd
# All right reserved.
#
# This file is confidential and NOT open source.  Do not distribute.
#

"""

"""

from generalized_backprop import make_problem, sigmoid
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

np.random.seed(42)


def _get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)

layers = [2, 3, 2, 2]

X, y = make_problem()
X = StandardScaler().fit_transform(X)

w1 = _get_weights((3, 2))
b1 = _get_weights((3, 1))

w2 = _get_weights((2, 3))
b2 = _get_weights((2, 1))

w3 = _get_weights((2, 2))
b3 = _get_weights((2, 1))

alpha = 0.3
N_ITER = 1000000


def run(w1, w2, w3, b1, b2, b3):
    losses = []
    for i in tqdm(xrange(N_ITER)):
        a1 = X.T
        z2 = np.dot(w1, a1) + b1
        a2 = sigmoid(z2)
        z3 = np.dot(w2, a2) + b2
        a3 = sigmoid(z3)
        z4 = np.dot(w3, a3) + b3
        a4 = sigmoid(z4)

        loss = ((a4 - y.T) ** 2).sum()
        # print loss
        losses.append(loss)

        # backprop
        del4 = -(y.T - a4) * a4 * (1 - a4)
        del3 = np.dot(w3.T, del4) * a3 * (1 - a3)
        del2 = np.dot(w2.T, del3) * a2 * (1 - a2)

        gradw3 = np.dot(del4, a3.T)
        gradb3 = del4.sum(1).reshape(-1, 1)

        gradw2 = np.dot(del3, a2.T)
        gradb2 = del3.sum(1).reshape(-1, 1)

        gradw1 = np.dot(del2, a1.T)
        gradb1 = del2.sum(1).reshape(-1, 1)

        w1 -= alpha * gradw1
        w2 -= alpha * gradw2
        w3 -= alpha * gradw3
        b1 -= alpha * gradb1
        b2 -= alpha * gradb2
        b3 -= alpha * gradb3

    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    run(w1, w2, w3, b1, b2, b3)
