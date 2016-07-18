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
from backprop import Layer

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
target = [0, 1, 1, 0]
alpha = 0.3

layers = [Layer(2, "input"),
          Layer(2, "hidden", theta=np.random.rand(3, 2)),
          Layer(1, "output", theta=np.random.rand(3,))]


class MultilayerPerceptron(object):

    def __init__(self, layers, alpha=0.3):
        self._layers = []
        for i, n_nodes in enumerate(layers):
            if i == 0:
                self._layers.append(Layer(n_nodes, kind="input"))
            else:
                theta = np.random.rand(layers[i - 1] + 1, n_nodes)
                if i == len(layers) - 1:
                    kind = "output"
                else:
                    kind = "hidden"
                self._layers.append(Layer(n_nodes, kind=kind, theta=theta))

    def _get_correction(self, layer, target):
        prediction = layer.output
        activation = layer.input
        multiplier = prediction * (1 - prediction) * activation
        if layer.kind == "output":
            delk = target - prediction
        else:
            delk = 0
        return delk * multiplier

    def learn_epoch(self, X, target):
        for layer in self._layers:
            layer.theta += self.alpha * self._get_correction(layer, target)

    def predict(self, X):
        l_input = X
        for layer in self._layers:
            l_input = layer.fit(l_input)
        return l_input

if __name__ == '__main__':
    mlp = MultilayerPerceptron(layers=[2, 2, 1])
    print mlp.predict(X)
    from IPython.core.debugger import Tracer
    Tracer()()
