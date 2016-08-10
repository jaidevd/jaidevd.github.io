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
Logistic Regression with Theano
"""


import theano
import theano.tensor as T
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, recall_score
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt


class LogisticRegression(object):

    def __init__(self, n_iter=100):
        self._x = T.dmatrix('x')
        self._y = T.dvector('y')
        self.n_iter = n_iter

    def _pos_prob(self):
        return

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.weights = theano.shared(np.random.random((X.shape[1],)), name="w")
        self.bias = theano.shared(np.random.random(), name="b")
        p_pos = 1 / (1.0 + T.exp(-T.dot(self._x, self.weights) - self.bias))
        predictionvar = p_pos > 0.5
        xentvar = -self._y * T.log(p_pos) - (1 - self._y) * T.log(1 - p_pos)
        costvar = xentvar.mean() + 0.01 * (self.weights ** 2).sum()

        gw, gb = T.grad(costvar, [self.weights, self.bias])

        # functions to be exposed to the user
        self._cost = theano.function([self._x, self._y], costvar)
        self._predict = theano.function([self._x], predictionvar)
        updates = (self.weights, self.weights - 0.1 * gw), \
                  (self.bias, self.bias - 0.1 * gb)  # training updates
        self._fit = theano.function([self._x, self._y],
                                    [predictionvar, xentvar],
                                    updates=updates)
        for i in range(self.n_iter):
            self._fit(X, y)

    def cost(self, X, y):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._cost(X, y)

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._predict(X)

if __name__ == '__main__':
    X, y = make_blobs(centers=2, cluster_std=3.0)
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()
    lr = LogisticRegression()
    for trainix, testix in StratifiedKFold(y, n_folds=5, shuffle=True):
        xTrain, yTrain = X[trainix, :], y[trainix]
        xTest, yTest = X[testix, :], y[testix]
        lr.fit(xTrain, yTrain)
        pred = lr.predict(xTest)
        print accuracy_score(yTest, pred), recall_score(yTest, pred)
