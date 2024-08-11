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
# from icecream import ic

pioneer = Pioneer(ip="10.1.100.104", mavlink_port=5656, logger=False)
drone = Drone(drone=pioneer, apply=True, joystick_on=True)
drone.arm()
# time.sleep(4)
drone.takeoff(1.8)
drone.set_coord_check()
drone.set_v()
drone.xyz_flag = True
drone.set_coord_check()
