from swarmsys import *
from ThreeDTool import Points
from pioneer_sdk import Pioneer
import matplotlib
import time
from loguru import logger

matplotlib.use('WebAgg')

pioners = []
ports = np.array([])

for i in range(9):
    pioners = np.hstack([pioners, Pioneer(ip="127.0.0.1", mavlink_port=8000 + i)])

rot_point = [0, 0, 0]
darr1 = Darray(apply=True)
darr1.create_local_array(pio_drones=pioners[0:9])

# darr2 = Darray(apply=True)
# darr2.create_local_array(pio_drones=pioners[4:8])


darr1.arm()
darr1.takeoff()
logger.debug(pioners[0].get_local_position_lps())
# darr2.arm()
# darr2.takeoff()

traj1 = [[0, 2.5, 1], [2.5, 2.5, 1], [2.5, 0, 1], [-2.5, -2.5, 1]]
#
darr1.go_traj(traj1)
time.sleep(5)
# logger.debug(pioners[0].get_local_position_lps())
for i in range(3):
    darr1.rot_z(np.pi/10*i, rot_point=rot_point, apply=True)
    darr1.wait_for_point()
