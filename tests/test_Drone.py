from swarmsys import *
from dspl import Dspl
import numpy as np
from body import Body
from ThreeDTool import Points

point = np.array([0, 1, 0])
ori = np.eye(3)

# drone = Drone(point, ori)
# drone2 = Drone(point, ori)
# drone2.rot_z(np.pi/4, apply=True)
#
# drone3 = Drone(point, ori)
# drone3.rot_z(np.pi/4*2, apply=True)
#
# drone4 = Drone(point, ori)
# drone4.rot_z(np.pi/4*3, apply=True)

drones = []
for i in range(8):
    drones.append(Drone(point, ori))
    drones[-1].rot_z(np.pi/4*i, apply=True)
point_rot = Points([[0, 0, 0]], text=True, s=10)
drones.append(point_rot)
dp = Dspl(drones, create_subplot=True, qt=True)
# drone.rot_x(np.pi/4, apply=True)
# drone.self_show()
dp.show()
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
