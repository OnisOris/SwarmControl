from swarmsys import *
from dspl import Dspl
import numpy as np
from ThreeDTool import Points
import matplotlib.pyplot as plt

dr = Drone()
rot_point = np.array([10, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)
dr.rot_z(pi/4, rot_point)

dp = Dspl([dr] + [p], qt=True)

lim_1 = -10
lim_2 = 10
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
dp.ax.quiver(*rot_point, 0, 0, 1,
                      length=10, color='orange')
dp.ax.title.set_text(f'Вращение дрона по оси z и точки {rot_point}')
plt.legend(['Ось вращения'])

dp.show()
