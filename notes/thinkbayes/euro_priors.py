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
import matplotlib.pyplot as plt
from thinkbayes import Suite


class Euro(Suite):

    def Likelihood(self, data, hypo):
        x = hypo
        if data == "H":
            return x / 100.0
        else:
            return 1 - x / 100.0


def update(suite):
    for i in range(140):
        suite.Update('H')
    for i in range(110):
        suite.Update('T')


# Uniform prior
uniform_hx = np.arange(101) / 101.0
uniform_euro = Euro(uniform_hx)
update(uniform_euro)

# Triangular prior
triangular_euro = Euro()
for i in range(51):
    triangular_euro.Set(i, i)
for i in range(51, 101):
    triangular_euro.Set(i, 100 - i)
update(triangular_euro)

plt.subplot(211)
plt.plot(np.arange(101), uniform_hx, "r", label="uniform")
plt.plot(np.arange(101),
        [i for i in range(51)] + [100 - i for i in range(51, 101)], "g",
        label="triangular")
plt.legend()

plt.subplot(212)
plt.plot(np.arange(101), uniform_euro.Probs(uniform_hx), "r", label="uniform")
plt.plot(np.arange(101), triangular_euro.Probs(np.arange(101)), "g",
        label="triangular")
plt.legend()
plt.show()
