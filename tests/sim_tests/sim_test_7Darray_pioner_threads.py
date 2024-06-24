from swarmsys import *
from pioneer_sdk import Pioneer
import matplotlib
import threading

matplotlib.use('WebAgg')

print(set_barycenter([[1, 1, 1], [2, 2, 2]]))
pioners = []
ports = np.array([])

for i in range(28):
    pioners = np.hstack([pioners, Pioneer(ip="127.0.0.1", mavlink_port=8000 + i)])
pioners = pioners.reshape((7, 4))

darray_arr = []

points = np.array([[-4, 4, 1], [0, 4, 1], [4, 4, 1],
                   [-4, 0, 1], [0, 0, 1], [4, 0, 1],
                   [0, -4, 1]])

for i in range(7):
    darr = Darray(apply=True)
    darr.create_local_array(pio_drones=pioners[i])
    darray_arr.append(darr)

for darr in darray_arr:
    t1 = threading.Thread(target=darr.arm, args=())
    t1.start()
    t1.join()

    t2 = threading.Thread(target=darr.takeoff, args=())
    t2.start()
    t2.join()
for darr in darray_arr:
    t3 = threading.Thread(target=darr.fasten, args=())
    t3.start()
    t3.join()

for darr in darray_arr:
    t4 = threading.Thread(target=darr.rot_z, args=(np.pi/2, None, True))
    t4.start()
    t4.join()
