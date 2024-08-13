from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
import matplotlib
import time
from config import CONFIG

matplotlib.use('WebAgg')

num = CONFIG['standard_port']
drone = Pioneer(ip=f"{CONFIG['ip_3']}{CONFIG['num_drone']}", mavlink_port=CONFIG['standard_port'])
dr = Drone(drone=drone)
dr.apply = True
# print(drone.get_local_position_lps())

dr.arm()
dr.takeoff(1.5)
time.sleep(7)

# dr.wait_point = True

dr.set_coord_check()
dr.set_v()

dr.speed_change([-1, 1, 0])

# dr.goto([-3, -3, 1.5], apply=True)
# # dr.wait_for_point()
# dr.goto([3, 3, 1.5], apply=True)
# # # # #
# time.sleep(5)
# # #
# drone.land()
# # time.sleep(5)
# drone.disarm()


