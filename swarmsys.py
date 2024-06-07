import numpy as np
from numpy import cos, sin, pi
from body import Body
from dspl import Dspl
from ThreeDTool import Line_segment
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
                                                     [0, 0, 1]]),
                 drone=None):
        """
        В данной реализации дрон имеет координату point и вектор ориантации orintation в глобальной системе координат
        :param point:
        :param orientation:
        """
        self.point = point
        self.orientation = orientation
        self.body = Body(self.point, self.orientation)
        self.trajectory = np.array([])
        self.drone = drone

    def attach_body(self, body):
        self.body = body

    def goto(self, point: list | np.ndarray | None = None, orientation: list | np.ndarray | None = None):
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
        # if self.trajectory.shape[0] == 0:
        segment = Line_segment(point1=self.point, point2=point)
        segment.color = 'orange'
        self.trajectory = np.hstack((self.trajectory, segment))
        if orientation is not None:
            self.orientation = orientation
        if orientation is None:
            self.point = point
        self.apply_position()

    def apply_position(self) -> None:
        """
        Данная функция отправляет дронам изменившеюся ориентацию и позицию в orientation и в point
        :return: None
        """
        # Здесь будет код для перемещения дрона, например через piosdk
        self.body.orientation = self.orientation
        self.body.point = self.point
        if self.drone is not None:
            self.drone.go_to_local_point(self.point[0], self.point[1], self.point[2], yaw=0)
            # while not self.drone.point_reached():
            #     a = self.drone.get_local_position_lps()
            #     if a is not None:
            #         print(a)

    def euler_rotate(self, alpha: float, beta: float, gamma: float, apply: bool = False) -> None:
        self.rot_z(alpha)
        self.rot_x(beta)
        self.rot_z(gamma)
        if apply:
            self.apply_position()

    def rot_x(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси y.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_x(self.orientation, -angle)
        self.point = rot_x(self.point - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def rot_y(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси y.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_y(self.orientation, -angle)
        self.point = rot_y(self.point - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([1, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает дрон вокруг выбранного центра rot_point по оси z.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: list[float, int] | np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_z(self.orientation, -angle)
        self.point = rot_z(self.point - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def self_show(self):
        dp = Dspl([self.body], create_subplot=True, qt=True)
        self.body.show(dp.ax)
        dp.show()

    def show(self, ax):
        self.body.show(ax)

    def arm(self) -> None:
        self.drone.arm()
    def takeoff(self) -> None:
        self.drone.takeoff()


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
        self.length = 5
        # self.body = Body(self.point, self.orientation)

    def __getitem__(self, item):
        return self.drones[item]

    def create_square_array(self, sizes: list | np.ndarray = np.array([[-10, 10], [-10, 10]]),
                            number_of_drones: int = 4, center_point: np.ndarray = np.array([0, 0, 0]),
                            pio_drones=None) -> None:
        """
        Функция генерирует квадрат из дронов
        :param sizes: Размер массива по x и по y
        :type sizes: list or np.ndarray
        :param center_point: Центральная точка формирования массива дронов
        :type center_point: np.ndarray
        :return:
        """
        self.xyz = center_point
        x = np.linspace(sizes[0, 0], sizes[0, 1], number_of_drones) + center_point[0]
        y = np.linspace(sizes[1, 0], sizes[1, 1], number_of_drones) + center_point[1]
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
        if pio_drones is not None:
            for i, drone in enumerate(self.drones):
                drone.drone = pio_drones[i]

    def fasten(self) -> None:
        """
        Данная функция перемещает дроны в установленный скелет массива
        :return:
        """
        for drone in self.drones:
            drone.goto(drone.point)

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

    def rot_x(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси x.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_x(self.orientation, -angle)
        for drone in self.drones:
            drone.rot_x(angle, rot_point, apply)
        self.xyz = rot_x(self.xyz - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def rot_y(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси y.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_y(self.orientation, -angle)
        for drone in self.drones:
            drone.rot_y(angle, rot_point, apply)
        self.xyz = rot_y(self.xyz - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def rot_z(self, angle: float | int, rot_point: np.ndarray = np.array([0, 0, 0]), apply: bool = True) -> None:
        """
        Данная функция вращает массив дронов вокруг выбранного центра rot_point по оси z.
        :param angle: Угол поворота в радианах
        :type angle: float | int
        :param rot_point: Коодинаты оси поворота
        :type rot_point: np.ndarray[float, int]
        :param apply: Отправлять ли данные в дроны?
        :type apply: bool
        :return: None
        """
        self.orientation = rot_z(self.orientation, -angle)
        for drone in self.drones:
            drone.rot_z(angle, rot_point, apply)
        self.xyz = rot_z(self.xyz - rot_point, angle) + rot_point
        if apply:
            self.apply_position()

    def show(self, ax):
        for drone in self.drones:
            drone.show(ax)

        ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                  self.orientation[0, 0], self.orientation[0, 1], self.orientation[0, 2],
                  length=self.length, color='r')
        ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                  self.orientation[1, 0], self.orientation[1, 1], self.orientation[1, 2],
                  length=self.length, color='g')
        ax.quiver(self.xyz[0], self.xyz[1], self.xyz[2],
                  self.orientation[2, 0], self.orientation[2, 1], self.orientation[2, 2],
                  length=self.length, color='b')

    def self_show(self):
        dp = Dspl(self.drones, create_subplot=True, qt=True)
        for drone in self.drones:
            drone.show(dp.ax)
        dp.show()

    def arm(self):
        for drone in self.drones:
            drone.arm()

    def takeoff(self) -> None:
        for drone in self.drones:
            drone.takeoff()
