from SwarmControl.swarmsys import *
from SwarmControl.dspl import Dspl
import numpy as np
from ThreeDTool import Points, loxodrome

dr = Drone()
zero_point = np.array([0, 0, 0])
traj = loxodrome(R=10, step=1, point_n=[0, 0, 10])
p = Points([zero_point], text=True)
p2 = Points(traj.tolist(), method='plot')

dr.goto([3, 0, 0])
dr.goto([3, 5, 0])
dr.goto([3, 0, 4])

for point in traj:
    dr.goto(point)

dp = Dspl([dr] + [p] + [p2] + dr.trajectory.tolist(), qt=True)
dp.limits(x=[-5, 5], y=[-5, 5], z=[-5, 5])
dp.ax.title.set_text('Перемещение дрона по траектории локсодромы')
dp.show()
