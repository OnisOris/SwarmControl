from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
import time

# import icecream as ic

pioneer = Pioneer(ip="10.1.100.108", mavlink_port=5656, logger=False)
drone = Drone(drone=pioneer, apply=True)
drone.arm()
time.sleep(4)
drone.takeoff(1.8)


drone.set_coord_check()
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
while k < np.pi/2*6:
    # k += np.pi/2
    drone.speed_change([sin(k), cos(k)])
    # v.speed = [sin(k), cos(k)]
    # print(f"xyz = {v.xyz} \n speed = {v.speed}")
    time.sleep(5)
    k += np.pi / 2

