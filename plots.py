from plotBuilder import Plotter
from icecream import ic
import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')
mpl.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt

# with open("./plot_out/test_105.npy", 'rb') as f:
#     array = np.load(f)
# ic(array)
plotter = Plotter("plot_out/test_105.npy")
plotter.all_plot()
plotter.xy_plot()
plotter.xvx_t()
plotter.reg_plot()
plotter.dot_xy_plot()
plotter.x_dot()
plotter.y_dot()
plotter.xy_pred()

