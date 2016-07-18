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
from xor_layered import target, X
from backprop import sigmoid

np.random.seed(1)
alpha = 0.3

# initializing weights
theta1 = np.random.rand(3, 2)
theta2 = np.random.rand(3,)


# forward pass
l1_op = sigmoid(np.dot(np.c_[X, np.ones((4,))], theta1))
l2_op = sigmoid(np.dot(np.c_[l1_op, np.ones((l1_op.shape[0],))], theta2))
prediction = l2_op

# output correction
del_op = (prediction * (1 - prediction) * (target - prediction)).reshape(-1, 1)
op_act = np.c_[l1_op, np.ones((l1_op.shape[0], 1))]
theta2 += alpha * np.dot(del_op.T, op_act).ravel()

# hidden layer correction
del_hidden_layer = np.dot(l2_op * (1 - l2_op),
                          np.dot(del_op, theta2.reshape(1, -1)))
from IPython.core.debugger import Tracer
Tracer()()
