import time

from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
from config import CONFIG

# import icecream as ic
# from icecream import ic

pioneer = Pioneer(ip="10.1.100.101", mavlink_port=5656, logger=False)
drone = Drone(CONFIG, drone=pioneer, apply=True, joystick_on=False)
drone.arm()
# time.sleep(4)
drone.takeoff(1.8)
drone.xyz_flag = True
drone.set_coord_check()
drone.set_v()

time.sleep(10)

drone.save_data('state_count.csv')
