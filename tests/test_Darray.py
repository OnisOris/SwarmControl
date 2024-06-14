import loguru
from icecream import icecream
from numpy import pi
from swarmsys import *
from ThreeDTool import Points

rot_point = np.array([10, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)
arr = Darray()
icecream.ic()
arr.create_square_array()
arr.rot_z(-pi, rot_point, apply=True)
dp = Dspl([arr] + [p], qt=True)
dp.show()

for drone in arr:
    loguru.logger.debug(drone.body)