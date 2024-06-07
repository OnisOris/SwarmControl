from swarmsys import *
from ThreeDTool import Points
import dataclasses
from pioneer_sdk import Pioneer

rot_point = np.array([0, 0, 0])
zero_point = np.array([0, 0, 0])
p = Points([rot_point, zero_point], text=True)

pioners = []

for i in range(4):
    pioners.append(Pioneer(ip="127.0.0.1", mavlink_port=8000+i))


arr = Darray()

arr.create_square_array(number_of_drones=2, center_point=np.array([0, 0, 0]), pio_drones=pioners,
                        sizes=np.array([[-0.5, 0.5], [-0.5, 0.5]]))

arr.arm()
arr.takeoff()

arr.fasten()
import time
time.sleep(10)
arr.rot_z(np.pi/4, rot_point=rot_point, apply=True)
time.sleep(3)
arr.rot_z(np.pi/4, rot_point=rot_point, apply=True)

