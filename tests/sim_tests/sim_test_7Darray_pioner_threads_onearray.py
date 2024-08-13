from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
import matplotlib
import time
matplotlib.use('WebAgg')

pioners = []
ports = np.array([])

for i in range(8):
    pioners = np.hstack([pioners, Pioneer(ip="127.0.0.1", mavlink_port=8000 + i)])

darr = Darray(apply=True)
darr.create_local_array(pio_drones=pioners)

darr.arm()
darr.takeoff()
darr.fasten()

for i in range(6):
    darr.rot_z(np.pi/6, apply=True)
    time.sleep(5)
