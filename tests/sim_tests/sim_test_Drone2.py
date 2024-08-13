import dataclasses

from SwarmControl.swarmsys import *
import matplotlib
from loguru import logger
matplotlib.use('WebAgg')


@dataclasses.dataclass
class IpPort:
    ip: str
    port: int


class DroneConnectingData:
    drone0: IpPort = IpPort(ip="127.0.0.1", port=8000)
    drone1: IpPort = IpPort(ip="127.0.0.1", port=8001)
    drone2: IpPort = IpPort(ip="127.0.0.1", port=8002)
    drone3: IpPort = IpPort(ip="127.0.0.1", port=8003)


pioner = Pioneer(ip=DroneConnectingData.drone0.ip, mavlink_port=DroneConnectingData.drone0.port)
dr = Drone(drone=pioner, apply=True)
dr.arm()
dr.takeoff()
time.sleep(7)
logger.debug(dr.drone.get_local_position_lps())
dr.goto([-1, 0, 1], apply=True)
dr.wait_for_point([-1, 0, 1])
logger.debug(dr.drone.get_local_position_lps())
# time.sleep(9)
# dr.rot_v(np.pi / 4, apply=True)
# time.sleep(9)
# dr.rot_v(np.pi / 4, rot_point=[0, 0, 0], apply=True)
# time.sleep(9)
# dr.rot_v(np.pi / 4, rot_point=[0, 0, 0], apply=True)
# dp = Dspl([dr])
# lim_1 = -10
# lim_2 = 10
# dp.limits(x=[lim_1, lim_2], y=[lim_1, lim_2], z=[lim_1, lim_2])
# dr.show_trajectory(dp.ax)
# dp.show()
# pioner.go_to_local_points(0, 0, 1, 90)
