import math

from point import Point
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')


class Syscord:
    def __init__(self, objects):
        self.objects = objects
        self.ax = None
        self.sim_flag = True
        self.delta = 0.1

    def real_time_start_sim(self):
        fig, ax = plt.subplots(figsize=(12, 7))
        t0 = time.time()
        dt = self.delta
        plt.ion()
        while self.sim_flag:
            start = time.time()
            time.sleep(self.delta)
            phi = time.time() - t0
            A = 0.5
            x_ = np.sin(phi) * A * math.exp(phi / 100)
            y_ = np.cos(phi) * A * math.exp(phi / 100)
            for obj in self.objects:
                obj.set_speed(np.array([x_, y_, 0]))
                obj.move(dt)
                x = obj.coords[0][0]
                y = obj.coords[0][1]
                x_s = x + x_
                y_s = y + y_

                plt.quiver(x, y,
                          x_s, y_s,)
                plt.scatter(x=obj.coords[0][0], y=obj.coords[0][1])


                print(obj.coords)
                length = np.linalg.norm(obj.speed)


                # obj.move(dt)
            # points_arr = np.vstack([points_arr, point.coords[0]])

            plt.xlim(-20, 20)
            plt.ylim(-20, 20)

            # plt.plot(points_arr[-3:-1, 0], points_arr[-3:-1, 1])
            plt.draw()
            plt.pause(0.0001)
            plt.clf()
            dt = time.time() - start

    # def rule_control(self, input):

    def limits(self, x: list | np.ndarray = None,
               y: list | np.ndarray = None,
               z: list | np.ndarray = None) -> None:
        if x is not None:
            self.ax.set_xlim(x[0], x[1])

        if y is not None:
            self.ax.set_ylim(y[0], y[1])

        if z is not None:
            self.ax.set_zlim(z[0], z[1])
