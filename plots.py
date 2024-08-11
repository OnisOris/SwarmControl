from plotBuilder import Plotter
from icecream import ic
import numpy as np
import matplotlib as mpl
# mpl.use('Qt5Agg')
mpl.use('TkAgg')
import pandas as pd
import matplotlib.pyplot as plt
from ThreeDTool import Up, Dspl

# with open("./plot_out/test_105.npy", 'rb') as f:
#     array = np.load(f)
# ic(array)
# plotter = Plotter("plot_out/test_105.npy")
# plotter.all_plot()
# plotter.xy_plot()
# plotter.xvx_t()
# plotter.reg_plot()
# plotter.dot_xy_plot()
# plotter.x_dot()
# plotter.y_dot()
# plotter.xy_pred()

# plotter2 = Plotter("plot_out/test_108_zeros.npy")
# # plotter2.all_plot()
# plotter2.xy_plot()
# # plotter2.xvx_t()
# # plotter2.reg_plot()
# # plotter2.dot_xy_plot()
# # plotter2.x_dot()
# # plotter2.y_dot()
# # plotter2.xy_pred()

df = pd.read_csv('./plot_out/data.csv', index_col=0)
# df.plot(x='t')
# plt.show()
ic(df[['x', 'y', 'z']].to_numpy())

up = Up(df[['x', 'y', 'z']], 'plot')
up_zeros = Up([[0, 0, 0]])
up_not_zeros = Up([[0, 0, 0]])
for i, obj in enumerate(up):
    if i == 0:
        continue
    elif i == up.shape[-2]:
        break
    if np.allclose(obj, [0., 0., 0.], 1e-8):
        point = np.mean(np.vstack([up[i-1], up[i+1]]), axis=0)
        ic(point)
        up_zeros = np.vstack([up_zeros, point])
    else:
        up_not_zeros = np.vstack([up_not_zeros, obj])

# up = Up([[0, 0, 10000], [1, 1.5, 0.2], [0.4, 0.2, 1], [3, 3, 3], [20, 0, 0]])
up_zeros = Up(up_zeros)
up_not_zeros = Up(up_not_zeros, 'plot')
up_zeros.color="red"
up_zeros.s = 50
dp = Dspl([up_not_zeros, up_zeros])
dp.show()

