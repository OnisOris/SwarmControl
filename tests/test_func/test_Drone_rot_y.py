from swarmsys import *
from dspl import Dspl
import numpy as np
from body import Body
from ThreeDTool import Points

dr = Drone()
rot_point = np.array([10, 10, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)
dr.rot_y(pi/4, rot_point)

dp = Dspl([dr] + [p], qt=True)
dp.show()
