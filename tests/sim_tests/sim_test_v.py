import dataclasses
from SwarmControl.swarmsys import *
import matplotlib

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
time.sleep(10)
# dr.v([1, 0, 0, 0.5])
k = 0
while True:
    k += np.pi/9
    dr.send_v([sin(k), cos(k), 0, 0])
    time.sleep(0.5)

