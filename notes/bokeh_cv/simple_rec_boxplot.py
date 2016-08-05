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

from bokeh.charts import BoxPlot
from bokeh.plotting import show, output_file
import numpy as np
import pandas as pd


x = np.random.rand(10, 3)
x[:, 2] = 1
df = pd.DataFrame(x, columns="accuracy recall n_iter".split())
output_file("boxplot.html")
p = BoxPlot(df, values="accuracy", label="n_iter", xscale="linear")
for i in range(5):
    x = np.random.rand(10, 3)
    x[:, 2] = df['n_iter'].max() + 1
    xdf = pd.DataFrame(x, columns="accuracy recall n_iter".split())
    df = pd.concat((df, xdf), axis=0)
    df.index = np.arange(df.shape[0])
    p = BoxPlot(df, values="accuracy", label="n_iter", xscale="linear")
show(p)
