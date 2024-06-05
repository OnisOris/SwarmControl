import numpy as np
from numpy import cos, sin, pi
from body import Body
from dspl import Dspl
from loguru import logger


def rot_x(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    rotate_x = np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])
    rot_vector = rotate_x.dot(vector)
    return rot_vector


def rot_y(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    rotate_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])
    rot_vector = rotate_y.dot(vector)
    return rot_vector


def rot_z(vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    rotate_z = np.array([[cos(angle), -sin(angle), 0],
                         [sin(angle), cos(angle), 0],
                         [0, 0, 1]])
    rot_vector = rotate_z.dot(vector)
    return rot_vector


def rot_v(axis: list | np.ndarray, vector: list | np.ndarray, angle: float | int) -> np.ndarray:
    """
    Функция вращает входные вектора вокруг произвольной оси, заданной векторами-столбцами
    :param axis: Вращаемые вектора
    :param vector:
    :param angle:
    :return:
    """
    x, y, z = axis
    rotate = np.array([[cos(angle) + (1 - cos(angle)) * x**2, (1-cos(angle) * x * y) -
                        sin(angle)*z, (1 - cos(angle)) * x * z + sin(angle) * y],
                       [(1 - cos(angle)) * y * x + sin(angle)*z, cos(angle) +
                        (1 - cos(angle)) * y**2, (1 - cos(angle)) * y * z - sin(angle) * x],
                       [(1 - cos(angle)) * z * x - sin(angle) * y, (1 - cos(angle)) * z * y +
                        sin(angle) * x, cos(angle) + (1 - cos(angle)) * z**2]])
    rot_vector = rotate.dot(vector)
    return rot_vector


class Drone:
    """
    Класс единицы дрона в трехмерном пространстве.
    """

    def __init__(self, point: np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]])):
        """
        В данной реализации дрон имеет координату point и вектор ориантации orintation в глобальной системе координат
        :param point:
        :param orientation:
        """
        self.point = point
        self.orientation = orientation
        self.body = Body(self.point, self.orientation)

    def attach_body(self, body):
        self.body = body

    def goto(self, point: None = None | list | np.ndarray, orientation: None = None | list | np.ndarray):
        """
        Функция перемещает дрон в заданное положение. Если задана точка назначения и вектор ориентации, то дрон поменяет
        то и то. Задана только ориентация или только точка, то изменится только нужный параметр. Если не задано ничего,
        то ничего не поменяется.
        :param orientation:
        :type orientation: list or np.ndarray or None
        :param point: Точка назначения для дрона
        :type point: list[float, int] or None
        :return:
        """
        if point is None and orientation is None:
            return
        # Здесь будет код для перемещения дрона, например через piosdk

        self.orientation = orientation
        self.point = point

    def apply_position(self) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        # Здесь будет код для перемещения дрона, например через piosdk
        self.body.orientation = self.orientation
        self.body.point = self.point

    def euler_rotate(self, alpha: float, beta: float, gamma: float, apply: bool = False) -> None:
        self.rot_z(alpha)
        self.rot_x(beta)
        self.rot_z(gamma)
        if apply:
            self.apply_position()

    def rot_x(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_x(self.orientation, -angle)
        self.point = rot_x(self.point, angle)
        if apply:
            self.apply_position()

    def rot_y(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_y(self.orientation, -angle)
        self.point = rot_y(self.point, angle)
        if apply:
            self.apply_position()

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
        self.orientation = rot_z(self.orientation, -angle)
        self.point = rot_z(self.point - rot_point, angle) + rot_z(self.point, angle)
        if apply:
            self.apply_position()
    # def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
    #     self.orientation = rot_z(self.orientation, angle)
    #     for drone in self.drones:
    #         drone.orientation = rot_z(drone.orientation - rot_point, angle) + rot_point
    #     if apply:
    #         self.apply_position()

    def self_show(self):
        dp = Dspl([self.body], create_subplot=True, qt=True)
        self.body.show(dp.ax)
        dp.show()

    def show(self, ax):
        self.body.show(ax)


class Darray:
    """
    Данный класс хранит в себе массив с дронами и имеет методы для их общего управления
    """

    def __init__(self, drones: list[Drone] | np.ndarray[Drone] = None,
                 xyz: list | np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                     [0, 1, 0],
                                                     [0, 0, 1]])):
        self.drones = drones
        self.xyz = xyz  # Координата массива дронов во внешней системе координат
        self.orientation = orientation  # Вектор ориентации системы коорднат массива дронов

    def __getitem__(self, item):
        return self.drones[item]

    def create_square_array(self, sizes: list | np.ndarray = np.array([[-10, 10], [-10, 10]]),
                            number_of_drones: int = 10) -> None:
        """
        Функция генерирует квадрат из дронов
        :param length: Длина квадрата
        :type length: float
        :param width: Ширина квадрата
        :type width: float
        :return:
        """

        x = np.linspace(sizes[0, 0], sizes[0, 1], number_of_drones)
        y = np.linspace(sizes[1, 0], sizes[1, 1], number_of_drones)
        z = np.ones(number_of_drones ** 2)
        x_m, y_m = np.meshgrid(x, y)
        x_m = x_m.reshape(-1)
        y_m = y_m.reshape(-1)
        points = np.vstack((x_m, y_m, z)).T
        # logger.debug(points)
        drones = np.array([])
        for point in points:
            drones = np.hstack((drones, Drone(point, self.orientation)))
        self.drones = drones

    def fasten(self) -> None:
        """
        Данная функция перемещает дроны в установленный скелет массива
        :return:
        """
        pass

    def unfasten(self) -> None:
        """
        Функция раскрепляет дроны друг от друга
        :return:
        """
        pass

    def apply_position(self) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        # Здесь будет код для перемещения дрона, например через piosdk
        pass

    def euler_rotate(self, alpha: float, beta: float, gamma: float, apply: bool = False) -> None:
        self.rot_z(alpha)
        self.rot_x(beta)
        self.rot_z(gamma)
        if apply:
            self.apply_position()

    def rot_x(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_x(self.orientation, angle)
        if apply:
            self.apply_position()

    def rot_y(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_y(self.orientation, angle)
        if apply:
            self.apply_position()

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = False) -> None:
        self.orientation = rot_z(self.orientation, angle)
        for drone in self.drones:
            drone.orientation = rot_z(drone.orientation - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def show(self, ax):
        for drone in self.drones:
            drone.show(ax)

    def self_show(self):
        dp = Dspl(self.drones, create_subplot=True, qt=True)
        for drone in self.drones:
            drone.show(dp.ax)
        dp.show()
