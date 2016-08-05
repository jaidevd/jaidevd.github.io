from bokeh.charts import BoxPlot, output_file, show
from bokeh.models.ranges import Range1d
from bokeh.charts.builders.boxplot_builder import BoxPlotBuilder
import numpy as np
import pandas as pd


def add_dummy_data(p, df):
    x = np.random.rand(10, 3)
    x[:, 2] = df['n_iter'].max() + 1
    xdf = pd.DataFrame(x, columns="accuracy recall n_iter".split())
    df = pd.concat((df, xdf), axis=0)
    p.add_builder(BoxPlotBuilder(df, values="accuracy", label="n_iter"))
    return df

x = np.random.rand(10, 3)
x[:, 2] = 1
df = pd.DataFrame(x, columns="accuracy recall n_iter".split())
# make simple dataframe
p = BoxPlot(df, values='accuracy', label='n_iter', x_range=Range1d(0, 1000))

for i in range(5):
    df = add_dummy_data(p, df)
print df

output_file("boxplot.html")

show(p)
