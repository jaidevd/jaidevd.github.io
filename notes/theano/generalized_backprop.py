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

import numpy as np
from theano import shared, function
import theano.tensor as T


def _get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)


class Backpropagation(object):

    def __init__(self, layers, alpha=0.3):
        self.alpha = alpha
        self.layers = layers
        self.weights = []
        self.biases = []
        self._x = T.dmatrix('x')
        self._y = T.dmatrix('y')
        for i, n in enumerate(layers):
            if i != len(layers) - 1:
                w = shared(_get_weights((layers[i + 1], n)),
                           name="w{}".format(i))
                b = shared(_get_weights((layers[i + 1], 1)),
                           name="b{}".format(i))
                self.weights.append(w)
                self.biases.append(b)

    def predict(self, X):
        self.layer_activations = [self._x.T]
        for i, weight in enumerate(self.weights):
            ai = self.layer_activations[i]
            activation = T.dot(weight, ai) + \
                self.biases[i].repeat(ai.shape[1], axis=1)
            self.layer_activations.append(1.0 / (1 + T.exp(-activation)))
        self._predict = function([self._x], [self.layer_activations[-1]])
        return self._predict(X)[0].T

    def fit(self, X, y, n_iter=1000, showloss=True):
        if showloss:
            self.losses = []
        self.predict(X)
        loss = T.sum((self.layer_activations[-1] - self._y.T) ** 2)
        updates = []
        for i in range(len(self.layers) - 1):
            w = self.weights[i]
            b = self.biases[i]
            grad_w, grad_b = T.grad(loss, [w, b])
            updates.append((w, w - self.alpha * grad_w))
            updates.append((b, b - self.alpha * grad_b))
        self._fit = function([self._x, self._y], [loss], updates=updates)
        for i in xrange(n_iter):
            print self._fit(X, y)

if __name__ == '__main__':
    xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    yTrain = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    bp = Backpropagation(layers=[2, 3, 2])
    bp.fit(xTrain, yTrain, n_iter=1000000)
