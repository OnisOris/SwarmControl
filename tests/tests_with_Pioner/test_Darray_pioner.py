from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
# from swarmsys import Darray
import matplotlib
matplotlib.use('WebAgg')
rot_point = np.array([0, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)

pioners = []

for i in range(4):
    pioners.append(Pioneer(ip="127.0.0.1", mavlink_port=8000+i))


arr = Darray(apply=True)

arr.create_square_array(number_of_drones=2, center_point=np.array([0, 0, 0]), pio_drones=pioners,
                        sizes=np.array([[-1, 1], [-1, 1]]))
arr.arm()
arr.takeoff()

arr.fasten()
import time
time.sleep(7)
for i in range(5):
    arr.rot_z(np.pi/10*i, rot_point=rot_point, apply=True)
    time.sleep(8)

dp = Dspl([], qt=False)
lim_1 = -1
lim_2 = 2
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
arr.show_trajectory(dp.ax)
dp.show()