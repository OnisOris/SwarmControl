import numpy as np


class Body:
    def __init__(self, point: np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]])):
        """
        :param point: Текущая координата тела
        :param orientation: Текущая ориентация тела. Она задается с помощью правой тройкой векторов
        """
        if not np.allclose(np.cross(orientation[0], orientation[1]), orientation[2], 1e-8):
            raise ValueError("Задана не правая единичная тройка векторов")

        if isinstance(point, list):
            self.point = np.array(point)
        else:
            self.point = point

        if isinstance(orientation, list):
            self.orientation = np.array(orientation)
        else:
            self.orientation = orientation
        self.name_body = None
        # Текущая координата тела
        # Матрица ориентации
        self.length = 0.5
        self.text = False


    def show(self, ax) -> None:
        """
        Функция отображает ориентацию тела в глобальной системе координат
        :param ax:
        :type ax: matplotlib.axes.Axes
        :return: None
        """
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
