from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
from config import CONFIG

pioneer = Pioneer(ip="10.1.100.104", mavlink_port=5656, logger=False)
drone = Drone(CONFIG, drone=pioneer, apply=True, joystick_on=True)
drone.arm()
drone.speed_flag = True
drone.xyz_flag = True
drone.takeoff(1.8)
drone.set_v()
drone.set_coord_check()


drone.set_coord_check()
