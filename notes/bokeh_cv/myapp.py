import numpy as np
import pandas as pd
from bokeh.plotting import figure, curdoc
from bokeh.driving import count


p = figure(x_range=[1, 20], y_range=(0, 1))
acc_line = p.line(x=[], y=[], legend="accuracy", color="red")
rec_line = p.line(x=[], y=[], legend="recall", color="yellowgreen")
acc_circle = p.circle(x=[], y=[], legend="accuracy", fill_color="red")
rec_circle = p.circle(x=[], y=[], legend="recall", fill_color="yellowgreen")

acc_line_ds = acc_line.data_source
rec_line_ds = rec_line.data_source
acc_circle_ds = acc_circle.data_source
rec_circle_ds = rec_circle.data_source

i = 0
curdoc().add_root(p)


@count()
def callback(i):
    x = np.random.rand(10, 2)
    df = pd.DataFrame(x, columns="accuracy recall".split())
    accuracy = x[:, 0].mean()
    recall = x[:, 1].mean()

    # accuracy data
    acc_data = {}
    acc_data['x'] = acc_line_ds.data['x'] + [i]
    acc_data['y'] = acc_line_ds.data['y'] + [accuracy]
    acc_line_ds.data = acc_data
    acc_circle_ds.data = acc_data

    # recall data
    rec_data = {}
    rec_data['x'] = rec_line_ds.data['x'] + [i]
    rec_data['y'] = rec_line_ds.data['y'] + [recall]
    rec_line_ds.data = rec_data
    rec_circle_ds.data = rec_data
    i += 1

curdoc().add_periodic_callback(callback, 1000)
