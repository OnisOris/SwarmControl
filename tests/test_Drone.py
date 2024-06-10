from swarmsys import *
from dspl import Dspl
import numpy as np
from ThreeDTool import Points
point = np.array([0, 1, 0])
ori = np.eye(3)
rot_point = np.array([1, 0, 0])
zero_point = np.array([0, 0, 0])
drones = []
for i in range(8):
    drones.append(Drone(point, ori))
    drones[-1].rot_z(np.pi/4*i, rot_point, apply=True)
point_rot = Points([rot_point, zero_point], text=True, s=10)
drones.append(point_rot)
dp = Dspl(drones, create_subplot=True, qt=True)
lim_1 = -2
lim_2 = 2
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])

dp.show()