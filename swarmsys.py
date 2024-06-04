import numpy as np
from body import Body

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
    rotate_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]])
    rot_vector = rotate_z.dot(vector)
    return rot_vector

class Drone:
    """
    Класс единицы дрона в трехмерном пространстве.
    """

    def __init__(self, point: np.ndarray = np.array([0, 0, 0]),
                 orientation: np.ndarray = np.array([[1, 0, 0],
                                                    [0, 1, 0],
                                                    [0, 0, 1]]),
                 ax=None):
        """
        В данной реализации дрон имеет координату point и вектор ориантации orintation в глобальной системе координат
        :param point:
        :param orientation:
        """
        self.point = point
        self.orientation = orientation
        self.ax = ax
        self.body = Body(self.point, self.orientation, ax=self.ax)



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

    def rot_z(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_z(self.orientation, angle)
        if apply:
            self.apply_position()

    def show(self):
        self.body.show()


class Darray:
    """
    Данный класс хранит в себе массив с дронами и имеет методы для их общего управления
    """

    def __init__(self, drones: list[Drone] | np.ndarray[Drone],
                 xyz: list[0, 0, 0] | np.ndarray,
                 orientation: list[0, 0, 0] | np.ndarray):
        self.drones = drones
        self.xyz = xyz  # Координата дрона во внешней системе координат
        self.orientation = orientation  # Вектор ориентации системы коорднат массива дронов

    def __getitem__(self, item):
        return self.drones[item]

    def create_square_array(self, length: float, width: float) -> None:
        """
        Функция генерирует квадрат из дронов
        :param length: Длина квадрата
        :type length: float
        :param width: Ширина квадрата
        :type width: float
        :return:
        """

        pass

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

    def rot_z(self, angle: float | int, apply: bool = False) -> None:
        self.orientation = rot_z(self.orientation, angle)
        if apply:
            self.apply_position()
