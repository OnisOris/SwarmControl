from swarmsys import *
from ThreeDTool import Dspl
import numpy as np
dp = Dspl([], qt=True)
point = np.array([0, 0, 0])
ori = np.eye(3)

drone = Drone(point, ori, ax=dp.ax)
dp.input_array.append(drone)
drone.show()
#
# for i in range(4):
#     drones.append(Drone(point=[i, i, 0], orientation=[(-1)**i, i, 0]))
#
#
# dp.ax.axes.set_xlim3d(left=-5, right=5)
# dp.ax.axes.set_ylim3d(bottom=-5, top=5)
# dp.ax.axes.set_zlim3d(bottom=0, top=5)
# dp.show()
#
# for dr in drones:
#     dr.rot_z(np.pi/3)
# drone.rot_z(np.pi/4)
# dp.create_subplot3D()
# dp.ax.axes.set_xlim3d(left=-5, right=5)
# dp.ax.axes.set_ylim3d(bottom=-5, top=5)
# dp.ax.axes.set_zlim3d(bottom=0, top=5)
#
# dp.show()
