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

    def loss(self, sample, target):
        return np.abs(self.predict(sample)[0] - target) ** 2

    def fit(self, X, y, n_iter=10000, verbose=True):
        i = 0
        while i < n_iter:
            for j in range(X.shape[0]):
                sample, target = X[j, :], y[j]
                self._learn_sample(sample, target)
            if verbose:
                if i % 100 == 0:
                    print i, self.loss(sample, target)
            i += 1

    def _learn_sample(self, sample, target):
        prediction, l1_op, sample = self.predict(sample)
        delk = (target - prediction) * prediction * (1 - prediction)
        delw2 = self.alpha * delk * l1_op
        delin = delk * self.theta2
        delj = delin * l1_op * (1 - l1_op)
        delw1 = self.alpha * np.dot(delj.reshape(-1, 1),
                                    sample.reshape(1, -1))
        delw1 = delw1.T[:, :(sample.shape[0] - 1)]
        self.theta1 += delw1
        self.theta2 += delw2

    def predict(self, sample):
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
    from IPython.core.debugger import Tracer
    Tracer()()
