import ThreeDTool
from swarmsys import *

drone1 = Drone(point=[2.5, 2.5, 0])
drone2 = Drone(point=[4.5, 6.5, 0])

plan_point = [8, 11, 0]

darr = Darray(drones=[drone1, drone2])

darr.self_show()