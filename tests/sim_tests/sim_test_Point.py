import math

from point import Point
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')


def limits(ax, x: list | np.ndarray = None,
           y: list | np.ndarray = None,
           z: list | np.ndarray = None) -> None:
    if x is not None:
        ax.set_xlim(x[0], x[1])

    if y is not None:
        ax.set_ylim(y[0], y[1])

    if z is not None:
        ax.set_zlim(z[0], z[1])


point = Point([0, 0, 0], speed=np.array([1, 0, 0]))

t0, delta = time.time(), 0.0001
dt = delta
old_time = time.time()
plt.ion()
points_arr = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
while True:
    start = time.time()
    time.sleep(delta)

    phi = time.time() - old_time
    A = 0.5
    x_ = np.sin(phi)*A*math.exp(phi/100)
    y_ = np.cos(phi)*A*math.exp(phi/100)
    point.set_speed(np.array([x_, y_, 0]))
    point.move(dt)
    print(point.coords)
    # points_arr = np.vstack([points_arr, point.coords[0]])

    plt.xlim(-20, 20)
    plt.ylim(-20, 20)
    plt.scatter(x=point.coords[0][0], y=point.coords[0][1])
    # plt.plot(points_arr[-3:-1, 0], points_arr[-3:-1, 1])
    plt.draw()
    plt.pause(0.0001)
    plt.clf()
    dt = time.time() - start




# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# matplotlib.use('Qt5Agg')

# import matplotlib.pyplot as plt
# import numpy as np

# plt.ion()
# for i in range(50):
#     y = np.random.random([10,1])
#     plt.plot(y)
#     plt.draw()
#     plt.pause(0.0001)
#     plt.clf()