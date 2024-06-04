import numpy as np


class Body:
    def __init__(self, point: np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]]),
                 ax=None,
                 k=1):
        self.drone = None
        # self.trajectory = None
        self.ax = ax
        # Текущая координата тела
        self.point = np.array([0, 0, 0])
        # Матрица ориентации
        self.orientation = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 1]])
        self.length = 1
        self.k = k
        self.lim_x = np.array([-1, 1])*k
        self.lim_y = np.array([-1, 1])*k
        self.lim_z = np.array([-1, 1])*k


    def show(self) -> None:
        """
        Функция отображает траекторию
        :param ax:
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        self.ax.set_xlim(self.lim_x)
        self.ax.set_ylim(self.lim_y)
        self.ax.set_zlim(self.lim_z)
        self.ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[0, 0], self.orientation[0, 1], self.orientation[0, 2],
                      length=self.length, color='r')
        self.ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[1, 0], self.orientation[1, 1], self.orientation[1, 2],
                      length=self.length, color='b')
        self.ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[2, 0], self.orientation[2, 1], self.orientation[2, 2],
                  length=self.length, color='g')
