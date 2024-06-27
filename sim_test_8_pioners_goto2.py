from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
import matplotlib
import time
from threading import Thread
from loguru import logger

matplotlib.use('WebAgg')
time.sleep(10)
pioners = []
ports = np.array([])

for i in range(6):
    pioners = np.hstack([pioners, Pioneer(ip="127.0.0.1", mavlink_port=8006 + i)])

rot_point = [0, 0, 0]
# darr1 = Darray(apply=True)
# darr1.create_local_array(pio_drones=pioners[0:6])

darr2 = Darray(apply=True)
darr2.create_local_array(pio_drones=pioners[0:6])


# darr1.arm()
# darr1.takeoff()
darr2.arm()
darr2.takeoff()

from ThreeDTool import Curve
traj5 = np.array([[2.5, 0.56, 1], [-3.12, 0.56, 1]])
# logger.debug(*traj1)
#
# th1 = Thread(target=darr1.go_traj(traj1))
darr2.go_traj(traj5)
# darr1.go_traj(traj1)
# darr2.go_traj(traj5)
# th1.start()

# time.sleep(5)
# th2.join()
# th1.join()
# logger.debug(pioners[0].get_local_position_lps())
for i in range(3):
    darr2.rot_z(np.pi/10*i, rot_point=rot_point, apply=True)
    darr2.wait_for_point()
