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
Basic delta rule learning in theano
"""

import theano
import theano.tensor as T
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, recall_score
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt


class LinearNeuron(object):

    def __init__(self):
        self._x = T.dmatrix('x')
        self._y = T.dvector('y')

    def fit(self, X, y, alpha=0.3, n_iter=5):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self._weights = theano.shared(np.random.random((X.shape[1],)), name="w")
        self._bias = theano.shared(np.random.random(), name="b")
        activation = 1.0 / (1 + T.exp(-T.dot(self._x, self._weights) -
            self._bias))
        prediction = activation >= 0.5
        self._predict = theano.function([self._x], prediction)
        cost = T.sum((activation - self._y) ** 2) / 2.0
        score = theano.function([self._x, self._y], cost)  # NOQA
        self._cost = theano.function([self._x, self._y], cost)
        gw, gb = T.grad(cost, [self._weights, self._bias])
        updates = (self._weights, self._weights - alpha * gw), \
            (self._bias, self._bias - alpha * gb)
        train = theano.function([self._x, self._y], [prediction, cost],
                                updates=updates)
        for i in range(n_iter):
            train(X, y)
            # print score(X, y)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._predict(X)

if __name__ == '__main__':
    X, y = make_blobs(centers=2, cluster_std=2.0)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    perc = LinearNeuron()
    print "Starting loop"
    for trainix, testix in StratifiedKFold(y, n_folds=5, shuffle=True):
        xTrain, yTrain = X[trainix, :], y[trainix]
        xTest, yTest = X[testix, :], y[testix]
        perc.fit(xTrain, yTrain, 100)
        pred = perc.predict(xTest)
        print accuracy_score(yTest, pred), recall_score(yTest, pred)
