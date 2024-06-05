import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class Dspl:
    """
    Класс нужен для отображения различных объектов.
    У каждого добавляемого объекта должен быть реализован метод show(ax),
    где ax является объектов matplotlib.Axes.ax
    """
    def __init__(self, input_array: list | np.ndarray, create_subplot: bool = True, qt: bool = False):
        """
        :param input_array: Принимаемый список из отображаемых объектов
        :type input_array: list or ndarray
        """
        if qt:
            mpl.use('Qt5Agg')
        self.input_array = input_array
        self.fig = None
        self.ax = None
        if create_subplot:
            self.create_subplot3D()

    def create_subplot3D(self) -> None:
        """
        Создание 3D сцены
        :return: None
        """
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

    def show(self) -> None:
        """
        Функция отображает траекторию
        :return: None
        """
        for obj in self.input_array:
            obj.show(self.ax)
        plt.show()
