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
from sklearn.datasets import make_circles

np.random.seed(1)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class Layer(object):
    """
    1. ALL WEIGHTS ARE INCOMING WEIGHTS
    2. Attaching the bias is the current layer's responsibility."""

    def __init__(self, n_nodes, kind, theta=None, activation=sigmoid, biased=True,
                 name=None):
        kind = kind.lower()
        assert kind in "input output hidden".split(), "Invalid layer type"
        if (theta is None) and (kind != "input"):
            raise TypeError("Hidden layer must have incoming weights.")
        elif (kind == "input") and (theta is not None):
            raise TypeError("Input layer doesn't have incoming weights.")
        self.kind = kind
        self.n_nodes = n_nodes
        self.theta = theta
        self.activation = activation
        self.biased = biased
        self.name = name

    def fit(self, X):
        if self.kind == "input":
            self.input = self.output = X
        else:
            if self.biased:
                X = np.c_[X, np.ones((X.shape[0],))]
            self.input = X
            self.output = self.activation(np.dot(X, self.theta))
        return self.output


class MultilayerPerceptron(object):

    def __init__(self, layers, alpha=0.3, activation=sigmoid):
        self.layers = layers
        self.alpha = alpha
        self.activation = activation
        # initializing the network
        self.weights = []
        for i, n_neurons in enumerate(layers[:-1]):
            theta = np.random.rand(n_neurons + 1, layers[i + 1])
            self.weights.append(theta)

    def loss(self, sample, target):
        return np.sum(np.abs(self.predict_epoch(sample)[0] - target) ** 2)

    def fit(self, X, y, n_iter=10000, verbose=True):
        i = 0
        losses = []
        while i < n_iter:
            self._learn_epoch(X, y)
            if verbose:
                if i % 100 == 0:
                    loss = self.loss(X, y, False)
                    print i, loss
                    losses.append(loss)
            i += 1
        return losses

    def predict_epoch(self, X):
        l_input = np.c_[X, np.ones((X.shape[0], 1))]
        activations = []
        for i, theta in enumerate(self.weights):
            l_output = self.activation(np.dot(l_input, theta))
            if i != len(self.weights) - 1:
                l_input = np.c_[l_output, np.ones((l_output.shape[0], 1))]
                activations.append(l_input)
        activations.append(np.c_[X, np.ones((X.shape[0], 1))])
        return l_output.ravel(), activations

    def _get_correction(self, i, j, yi, yj, target):
        assert j - i == 1, "Invalid layer pair."
        if j == len(self.layers):
            return (yj - target) * yj * (1 - yi) * yi

    def _learn_epoch(self, X, target):
        # weights between 3rd and 4th layer
        prediction, activations = self.predict_epoch(X)
        # delk = (target - prediction) * prediction * (1 - prediction)
        # l3_activation = activations[-1]
        # l2_activation = activations[-2]
        # l1_activation = activations[-3]
        # delw3 = self.alpha * np.dot(delk, l3_activation)
        #
        # # weights between 2nd and 3rd layer
        # delin = np.dot(delk.reshape(-1, 1), self.weights[-1].reshape(1, -1))
        # delj = delin * l3_activation * l3_activation * (1 - l3_activation)
        # delw2 = self.alpha * np.dot(delj.T, l2_activation)
        # del2 = delw2[:(l2_activation.shape[1] - 1), :].T
        #
        # # weights between 1st and 2nd layer
        # delin = np.dot(delk.reshape(-1, 1), self.weights[-2].reshape(1, -1))
        # delj = delin * l2_activation * l2_activation * (1 - l2_activation)
        # delw1 = self.alpha * np.dot(delj.T, l1_activation)
        # delw1 = delw1[:(l1_activation.shape[1] - 1), :].T
        #
        # # updating
        # self.weights[0] += delw1
        # self.weights[1] += delw2
        # self.weights[2] += delw3
        return


class Backprop(object):

    def __init__(self, alpha=0.3, activation=sigmoid):
        self.theta1 = np.random.random((3, 2))
        self.theta2 = np.random.random((3,))
        self.alpha = alpha
        self.activation = activation

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
        losses = []
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
                    loss = self.loss(sample, target, online)
                    print i, loss
                    losses.append(loss)
            i += 1
        return losses

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
        l1_op = self.activation(np.dot(X, self.theta1))
        l2_activation = np.c_[l1_op, np.ones((l1_op.shape[0], 1))]
        prediction = self.activation(np.dot(l2_activation, self.theta2))
        return prediction, l2_activation, X

    def predict_sample(self, sample):
        sample = np.r_[sample, 1]
        l1_op = self.activation(np.dot(sample, self.theta1))
        l2_activation = np.r_[l1_op, 1]
        l2_op = self.activation(np.dot(l2_activation, self.theta2))
        return l2_op, l2_activation, sample


if __name__ == '__main__':
    X, y = make_circles(factor=0.1, noise=0.08)
    mlp = MultilayerPerceptron(layers=[2, 4, 2, 1])
    mlp.fit(X, y, n_iter=1)
