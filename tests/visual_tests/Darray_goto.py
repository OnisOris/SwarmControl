# for drone in dr
from swarmsys import *
from dspl import Dspl
import numpy as np
from ThreeDTool import Points, loxodrome

# rot_point = np.array([40, 0, 0])
zero_point = np.array([0, 0, 0])
darr = Darray()
darr.create_square_array(center_point=np.array([0, 0, 0]))

zero_point = np.array([0, 0, 0])
traj = loxodrome(R=10, step=1, point_n=[0, 0, 10])
p = Points([zero_point], text=True)
p2 = Points(traj.tolist(), method='plot')



darr.goto([3, 0, 0])
darr.goto([3, 5, 0])
darr.goto([3, 0, 4])

for point in traj:
    darr.goto(point)

dp = Dspl([darr] + [p] + [p2] + darr.trajectory.tolist(), qt=True)
dp.limits(x=[-5, 5], y=[-5, 5], z=[-5, 5])
dp.ax.title.set_text('Перемещение дронов по траектории локсодромы')
dp.show()
from loguru import logger
logger.debug(darr.body.point)
