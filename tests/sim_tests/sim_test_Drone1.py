import dataclasses
import time

from pioneer_sdk import Pioneer
from swarmsys import *


@dataclasses.dataclass
class IpPort:
    ip: str
    port: int


class DroneConnectingData:
    drone0: IpPort = IpPort(ip="127.0.0.1", port=8003)
    drone1: IpPort = IpPort(ip="127.0.0.1", port=8007)
    drone2: IpPort = IpPort(ip="127.0.0.1", port=8005)
    drone3: IpPort = IpPort(ip="127.0.0.1", port=8004)


pioner = Pioneer(ip=DroneConnectingData.drone0.ip, mavlink_port=DroneConnectingData.drone0.port)
dr = Drone(drone=pioner, apply=True)
dr.arm()
dr.takeoff()
dr.goto([1, -2, 1], apply=True)
n = False
from loguru import logger
while not n:
    k = pioner.get_local_position_lps()
    if isinstance(k, list):
        logger.debug(k)
        n = np.allclose(np.array(k), np.array([1, -2, 1]), 1e-1)

for i in range(8):
    dr.rot_z(np.pi / 4, apply=True)
    n = False
    while not n:
        k = pioner.get_local_position_lps()
        if isinstance(k, list):
            logger.debug(k)
            n = np.allclose(np.array(k), dr.body.point, 1e-1)
