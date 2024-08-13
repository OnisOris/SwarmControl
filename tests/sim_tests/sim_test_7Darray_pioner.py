from SwarmControl.swarmsys import *
from pioneer_sdk import Pioneer
import matplotlib

matplotlib.use('WebAgg')

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
    darr.create_square_array(number_of_drones=2, center_point=points[i], pio_drones=pioners[i],
                             sizes=np.array([[-0.5, 0.5], [-0.5, 0.5]]))
    darray_arr.append(darr)

for darr in darray_arr:
    darr.arm()
    darr.takeoff()
