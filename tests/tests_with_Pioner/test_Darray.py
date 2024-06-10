from swarmsys import *
from ThreeDTool import Points
import dataclasses
from pioneer_sdk import Pioneer

rot_point = np.array([1, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)

pioners = []

arr = Darray()

arr.create_square_array(number_of_drones=2, center_point=np.array([0, 0, 0]), pio_drones=None,
                        sizes=np.array([[-0.5, 0.5], [-0.5, 0.5]]))
arr.apply = False
arr.arm()
arr.takeoff()

arr.fasten()
for i in range(3):
    arr.rot_z(np.pi/4, rot_point=rot_point, apply=False)

dp = Dspl([], qt=True)
lim_1 = -1
lim_2 = 2
dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
arr.show_trajectory(dp.ax)
dp.show()