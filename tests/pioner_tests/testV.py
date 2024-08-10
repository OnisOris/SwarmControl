from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
import matplotlib
import time

matplotlib.use('WebAgg')

# pioner = Pioneer(ip="10.1.100.103", mavlink_port=5656)
    # IpPort(ip="10.1.100.103", port=5656)

points = np.array([[-4, 4, 1], [0, 4, 1], [4, 4, 1],
                   [-4, 0, 1], [0, 0, 1], [4, 0, 1],
                   [0, -4, 1]])
# pioner.arm()
# # time.sleep(2)
# pioner.takeoff()
# ip="10.1.100.103", port=5656
# (ip="127.0.0.1", port=8000)
drone = Pioneer(ip="10.1.100.106", mavlink_port=5656)
dr = Drone(drone=drone)
# print(drone.get_local_position_lps())

dr.arm()
dr.takeoff()
# # # # #
time.sleep(5)
# #
while True:
    dr.send_v([0.2, 0, 0, 0])
# drone.disarm()


