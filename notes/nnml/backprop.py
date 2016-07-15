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

np.random.seed(1)


class Backprop(object):

    def __init__(self, alpha=0.3):
        self.theta1 = np.random.random((3, 2))
        self.theta2 = np.random.random((3,))
        self.alpha = alpha

    def logit(self, x):
        return 1.0 / (1 + np.exp(-x))

    def loss(self, sample, target, online):
        if online:
            return np.abs(self.predict_sample(sample)[0] - target) ** 2
        else:
            return np.sum(np.abs(self.predict_epoch(sample)[0] - target) ** 2)

    def _learn(self, X, y):
        if X.ndim in (1, 2):
            if (1 in X.shape) or X.ndim == 1:
                X = X.ravel()
                assert(np.isscalar(y))
                return self._learn_sample(X, y)
        return self._learn_epoch(X, y)

    def fit(self, X, y, n_iter=10000, verbose=True, online=False):
        i = 0
        while i < n_iter:
            if online:
                for j in range(X.shape[0]):
                    sample, target = X[j, :], y[j]
                    self._learn_sample(sample, target)
            else:
                sample, target = X, y
                self._learn_epoch(sample, target)
            if verbose:
                if i % 100 == 0:
                    print i, self.loss(sample, target, online)
            i += 1

    def _learn_sample(self, sample, target):
        prediction, l1_op, sample = self.predict_sample(sample)
        delk = (target - prediction) * prediction * (1 - prediction)
        delw2 = self.alpha * delk * l1_op
        # delk for a sample is a scalar.
        # So for a en epoch, it should be a vector of size m
        # delin for a sample is a vector of size 3 (no of neurons
        # ( in the hidden layer)
        # So for an epoch, it should be a matrix of size m by s_{l}
        delin = delk * self.theta2
        # delj for a sample is a vector of size 3
        # Therefore for the epoch it should be of size m by s_{}
        delj = delin * l1_op * (1 - l1_op)
        # delw1 for a sample is a 3 by 3 matrix whose first two rows are used
        # therefore for an epoch it should be a structure that contains
        # m by 3 by 2 matrix
        delw1 = self.alpha * np.dot(delj.reshape(-1, 1),
                                    sample.reshape(1, -1))
        delw1 = delw1[:(sample.shape[0] - 1), :].T
        self.theta1 += delw1
        self.theta2 += delw2

    def _learn_epoch(self, X, target):
        prediction, l2_activation, X = self.predict_epoch(X)
        delk = (target - prediction) * prediction * (1 - prediction)
        delw2 = self.alpha * np.dot(delk, l2_activation)
        delin = np.dot(delk.reshape(-1, 1), self.theta2.reshape(1, -1))
        delj = delin * l2_activation * (1 - l2_activation)
        delw1 = self.alpha * np.dot(delj.T, X)
        delw1 = delw1[:(X.shape[1] - 1), :].T
        self.theta1 += delw1
        self.theta2 += delw2

    def predict_epoch(self, X):
        X = np.c_[X, np.ones((X.shape[0], 1))]
        l1_op = self.logit(np.dot(X, self.theta1))
        l2_activation = np.c_[l1_op, np.ones((l1_op.shape[0], 1))]
        prediction = self.logit(np.dot(l2_activation, self.theta2))
        return prediction, l2_activation, X

    def predict_sample(self, sample):
        sample = np.r_[sample, 1]
        l1_op = self.logit(np.dot(sample, self.theta1))
        l2_activation = np.r_[l1_op, 1]
        l2_op = self.logit(np.dot(l2_activation, self.theta2))
        return l2_op, l2_activation, sample


if __name__ == '__main__':
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    t = [0, 1, 1, 0]
    bp = Backprop()
    bp.fit(X, t, 1000000)
    print bp.predict_epoch(X)[0]
