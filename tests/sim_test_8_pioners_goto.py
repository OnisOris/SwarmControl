from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
import matplotlib

matplotlib.use('WebAgg')

pioners = []
ports = np.array([])

for i in range(6):
    pioners = np.hstack([pioners, Pioneer(ip="127.0.0.1", mavlink_port=8000 + i)])

rot_point = [0, 0, 0]
darr1 = Darray(apply=True)
darr1.create_local_array(pio_drones=pioners[0:6])

# darr2 = Darray(apply=True)
# darr2.create_local_array(pio_drones=pioners[6:12])


darr1.arm()
darr1.takeoff()
# darr2.arm()
# darr2.takeoff()

# [2.5, 0, 1], [-2.5, -2.5, 1]
traj1 = np.array([[-3.12, 1.66, 1], [-1, 1.66, 1]])
# logger.debug(*traj1)
# traj2 = traj1[::-1]
#
# th1 = Thread(target=darr1.go_traj(traj1))
# th2 = Thread(target=darr2.go_traj(traj2))
darr1.go_traj(traj1)

points_pl = np.array([[0, 1.3, 0], [0, -1.3, 0]])

darr1.drones[0].body.point = darr1.drones[0].body.point + points_pl[0]
darr1.drones[3].body.point = darr1.drones[3].body.point + points_pl[1]

darr1.drones[2].body.point = darr1.drones[2].body.point + points_pl[1]
darr1.drones[5].body.point = darr1.drones[5].body.point + points_pl[0]

darr1.fasten()
traj2 = np.array([[3.92, 1.66, 1]])
darr1.go_traj(traj2)

darr1.drones[0].body.point = darr1.drones[0].body.point - points_pl[0]
darr1.drones[3].body.point = darr1.drones[3].body.point - points_pl[1]

darr1.drones[2].body.point = darr1.drones[2].body.point - points_pl[1]
darr1.drones[5].body.point = darr1.drones[5].body.point - points_pl[0]
darr1.fasten()

# th2.start()
# th1.start()

# time.sleep(5)
# th2.join()
# th1.join()
# logger.debug(pioners[0].get_local_position_lps())
for i in range(3):
    darr1.rot_z(np.pi/10*i, rot_point=rot_point, apply=True)
    darr1.wait_for_point()
traj3 = [[-3, -3, 1]]
darr1.go_traj(traj3)
for i in range(2):
    darr1.rot_z(np.pi/10*i, rot_point=rot_point, apply=True)
    darr1.wait_for_point()