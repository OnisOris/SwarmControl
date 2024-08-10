from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
import matplotlib
import time
from config import CONFIG

matplotlib.use('WebAgg')

num = CONFIG['standard_port']
drone = Pioneer(ip=f"{CONFIG['ip_3']}{CONFIG['num_drone']}", mavlink_port=CONFIG['standard_port'])
dr = Drone(drone=drone)
# print(drone.get_local_position_lps())

dr.arm()
dr.takeoff()
# time.sleep(10)
dr.apply=True
dr.wait_point = True
dr.goto([-3, -3, 1.5], apply=True)
# dr.wait_for_point()
dr.goto([3, 3, 1.5], apply=True)
# # # # #
# time.sleep(5)
# # #
# drone.land()
# # time.sleep(5)
# drone.disarm()


