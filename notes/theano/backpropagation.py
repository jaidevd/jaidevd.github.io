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
from theano import tensor as T

alpha = 0.3


def _get_weights(shape):
    std = 1.0 / np.sqrt(shape[1] + 1)
    return np.random.normal(scale=std, size=shape)

x = T.dmatrix('x')
y = T.dmatrix('y')

w1 = shared(_get_weights((3, 2)), name="w1")
w2 = shared(_get_weights((2, 3)), name="w2")
b1 = shared(_get_weights((3, 1)), name="b1")
b2 = shared(_get_weights((2, 1)), name="b1")


a1 = x.T
z2 = T.dot(w1, a1) + b1.repeat(a1.shape[1], axis=1)
a2 = 1.0 / (1 + T.exp(-z2))
z3 = T.dot(w2, a2) + b2.repeat(a2.shape[1], axis=1)
a3 = 1.0 / (1 + T.exp(-z3))

predict = function([x], a3)


loss = T.sum((a3 - y.T) ** 2)

gradw1, gradb1 = T.grad(loss, [w1, b1])
gradw2, gradb2 = T.grad(loss, [w2, b2])

updates = [
        (w1, w1 - alpha * gradw1),
        (w2, w2 - alpha * gradw2),
        (b1, b1 - alpha * gradb1),
        (b2, b2 - alpha * gradb2)]


train = function([x, y], [loss], updates=updates)
xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
yTrain = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

for i in xrange(1000000):
    print i, train(xTrain, yTrain)
