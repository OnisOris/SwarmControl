import threading
import time

from pioneer_sdk import Pioneer
from numpy import pi, sin, cos
from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
import matplotlib
import time
from config import CONFIG
# import icecream as ic
from icecream import ic

pioneer = Pioneer(ip="10.1.100.105", mavlink_port=5656, logger=False)
drone = Drone(drone=pioneer, apply=True)
drone.set_coord_check()
drone.arm()
drone.takeoff(1.8)
time.sleep(3)
coord = [-2, -2, 1.5]
# coord = rot_z(coord, -pi / 2)
drone.goto(coord, apply=True)
drone.wait_for_point(coord, accuracy=1e-1)
drone.set_v()
drone.xyz_flag = True
# time.sleep(10)
# start_pos = np.array(dr.get_local_position_lps())

# time.sleep(5)
# v.set_v()
drone.set_coord_check()
t0 = time.time()
# flag = True
k = 0
# drone.write_traj = True
# print("fff")
while k < np.pi/2*3:
    # k += np.pi/2
    drone.speed_change([sin(k), cos(k)])
    # v.speed = [sin(k), cos(k)]
    # print(f"xyz = {v.xyz} \n speed = {v.speed}")
    time.sleep(5)
    k += np.pi / 2
drone.stop()
# v.flag = False
# v.xyz_flag = False
time.sleep(2)
drone.land()
time.sleep(8)
drone.disarm()
# # 70, -70
with open(f'../../plot_out/test_105.npy', 'wb') as f:
    np.save(f, drone.traj)
