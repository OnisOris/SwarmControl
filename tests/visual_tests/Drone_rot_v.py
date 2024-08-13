from SwarmControl.swarmsys import *
from SwarmControl.dspl import Dspl
import numpy as np
from ThreeDTool import Points
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

dr = Drone()
rot_point = np.array([3, 0, 0])
zero_point = np.array([0, 0, 0])
axis = np.array([1, 1, 1])
p = Points([rot_point, zero_point], text=True)
dr.rot_v(np.pi/4, rot_point, axis=axis)

dp = Dspl([dr] + [p])

lim_1 = -10
lim_2 = 10
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
dp.ax.quiver(*rot_point, *axis,
                      length=10, color='orange')
dp.ax.title.set_text(f'Вращение дрона по оси z и точки {rot_point}')
plt.legend(['Ось вращения'])

dp.show()
