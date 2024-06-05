import numpy as np


class Body:
    def __init__(self, point: np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]]),
                 k=1):
        self.name_body = None
        # Текущая координата тела
        self.point = point
        # Матрица ориентации
        self.orientation = orientation
        self.length = 1
        self.k = k
        self.lim_x = np.array([-1, 1])*k
        self.lim_y = np.array([-1, 1])*k
        self.lim_z = np.array([-1, 1])*k
        self.text = False


    def show(self, ax) -> None:
        """
        Функция отображает траекторию
        :param ax:
        :type ax: matplotlib.axes.Axes
        :return: None
        """
        ax.set_xlim(self.lim_x)
        ax.set_ylim(self.lim_y)
        ax.set_zlim(self.lim_z)
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[0, 0], self.orientation[0, 1], self.orientation[0, 2],
                      length=self.length, color='r')
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[1, 0], self.orientation[1, 1], self.orientation[1, 2],
                      length=self.length, color='g')
        ax.quiver(self.point[0], self.point[1], self.point[2],
                  self.orientation[2, 0], self.orientation[2, 1], self.orientation[2, 2],
                  length=self.length, color='b')
        if self.text:
            if self.name_body != None:
                ax.text(self.point[0], self.point[1], self.point[2],
                        f"{self.name_body}: \n {self.point[0], self.point[1], self.point[2]}", color='blue')
            else:
                ax.text(self.point[0], self.point[1], self.point[2],
                    f"{np.round(self.point[0], 3), np.round(self.point[1], 3), np.round(self.point[2], 3)}", color='blue')
